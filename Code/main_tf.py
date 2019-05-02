import sys
sys.path.insert(0, "./")
import tensorflow as tf
import basic_block
import time
import os
import load_data

@tf.function
def train(style_list, content_list, batch_size, num_epochs, style_weight, content_weight,
          ngf, log_interval, save_model_dir):

    ########################
    # Data loader
    ########################

    content_loader = load_data.get_dataloader(content_list, batch_size)
    style_loader = load_data.get_dataloader(style_list, batch_size)

    ########################
    # Init model
    ########################
    vgg = basic_block.Vgg()
    style_model = basic_block.Net(ngf)

    ########################
    # optimizer and loss
    ########################
    mse_loss = tf.keras.losses.mean_squared_error()
    optimizer = tf.keras.optimizers.Adam()

    ########################
    # Start training loop
    ########################
    for epoch in range(1, num_epochs):
        agg_content_loss = 0.0
        agg_style_loss = 0.0
        count = 0
        for batch_id, content_img in enumerate(content_loader):
            with tf.GradientTape() as tape:
                n_batch = len(content_img)
                count += n_batch
                # data preparation. TODO: figure out these helper functions
                style_image = next(style_loader)
                #style_v = utils.subtract_imagenet_mean_preprocess_batch(style_image.copy())

                feature_style = vgg(style_image)
                gram_style = [basic_block.gram_matrix(y) for y in feature_style]

                f_xc_c = vgg(content_img)[1]

                style_model.set_target(style_image)
                y = style_model(content_img)
                features_y = vgg(y)

                # TODO: why the coefficient 2?
                content_loss = 2 * content_weight * mse_loss(features_y[1], f_xc_c)

                style_loss = 0.0
                for m in range(len(features_y)):
                    gram_y = basic_block.gram_matrix(features_y[m])
                    _, C, _ = gram_style[m].shape
                    gram_s = tf.expand_dims(gram_style[m], 0).broadcast_to(batch_size, 1, C, C)
                    style_loss += 2 * style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])
                total_loss = content_loss + style_loss
                agg_content_loss += content_loss[0]
                agg_style_loss += style_loss[0]
            gradients = tape.gradient(total_loss, style_model.variables)
            optimizer.apply_gradients(zip(gradients, style_model.trainable_variables))

            if (batch_id + 1) % log_interval == 0:
                mesg = "{}\tEpoch {}:\tcontent: {:.3f}\tstyle: {:.3f}\ttotal: {:.3f}".format(
                    time.ctime(), epoch + 1,
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if (batch_id + 1) % (4 * log_interval) == 0:
                # save model
                save_model_filename = "Epoch_" + str(epoch) + "iters_" + \
                    str(count) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
                    content_weight) + "_" + str(style_weight) + ".params"
                save_model_path = os.path.join(save_model_dir, save_model_filename)
                tf.saved_model.save(style_model, save_model_path)
                print("\nCheckpoint, trained model saved at", save_model_path)

