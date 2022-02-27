from turtle import shape
import tensorflow as tf
from enhance.models import Generator, Discriminator
import wandb
import time
import matplotlib.pyplot as plt
import os


class Optimiser:
    def __init__(self,
                 checkpoint_dir,
                 image_size,
                 lr_generator,
                 lr_discriminator):

        self.generator= Generator(image_size=image_size)

        self.discriminator_model = Discriminator.get_model(image_size=image_size)
        self.discriminator_loss = Discriminator.loss

        self.generator_optimizer = tf.keras.optimizers.Adam(lr_generator, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            lr_discriminator, beta_1=0.5)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator.model,
                                              discriminator=self.discriminator_model)

    def generate_images(self, model, dataset, step, num_images=3):
        fig, axs = plt.subplots(num_images, 3)

        for k in range(num_images):
            example_input, example_target = next(iter(dataset.take(1)))

            prediction = model(example_input, training=True)

            display_list = [example_input[0], example_target[0], prediction[0]]
            title = ['Input Image', 'Ground Truth', 'Predicted Image']

            for i in range(3):
                #axs[k, i].set_title(title[i])
                # Getting the pixel values in the [0, 1] range to plot.
                axs[k, i].imshow(display_list[i] * 0.5 + 0.5)
                axs[k, i].axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()

        wandb.log({
            'train/epoch': step,
            'train/prediction': wandb.Image(plt)}
        )

        plt.close()


    def train(self, train_ds, val_ds, steps=10):
        start = time.time()

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if (step) % 1000 == 0:
                if step != 0:
                    print(
                        f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

                start = time.time()
                self.generate_images(self.generator.model,
                                     val_ds,
                                     step)

                print(f"Step: {step//1000}k")

            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.train_step(
                input_image, target)

            # Training step
            if (step+1) % 10 == 0:
                print('.', end='', flush=True)
                wandb.log({
                    "train/epoch": step,
                    "train/gen_total_loss": gen_total_loss.numpy(),
                    "train/gen_gan_loss": gen_gan_loss.numpy(),
                    "train/gen_l1_loss": gen_l1_loss.numpy(),
                    "train/disc_loss": disc_loss.numpy()
                })

            # Save (checkpoint) the model every 5k steps
            if (step + 1) % 5000 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        self.generator.save_model(self.checkpoint_dir)
        
    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator.model(input_image, training=True)

            disc_real_output = self.discriminator_model(
                [input_image, target], training=True)
            disc_generated_output = self.discriminator_model(
                [input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator.loss(
                disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(
                disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.model.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator_model.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.model.trainable_variables))

        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator_model.trainable_variables))

        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss
