import tensorflow as tf
import net
import time
import os

@tf.function
def train(args):

    ########################
    # Data loader
    ########################

    train_dataset = data.ImageFolder()
    train_loader = tf.DataLoader(train_dataset, batch_size = args.batch_size)
    style_loader = utils.StyleLoader()

    ########################
    # Init model
    ########################
    vgg = net.vgg16()
    utils.init_vgg_params(vgg)
    style_model = net.Net(ngf=args.ngf)

    ########################
    # optimizer and loss
    ########################
    mse_loss = tf.keras.losses.mean_squared_error()
    optimizer = tf.keras.optimizers.Adam()

    ########################
    # Start training loop
    ########################
    for e in range(args.epochs):
        agg_content_loss = 0.0
        agg_style_loss = 0.0
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            with tf.GradientTape() as tape:
                n_batch = len(x)
                count += n_batch
                # data preparation. TODO: figure out these helper functions
                style_image = style_loader.get(batch_id)
                style_v = utils.subtract_imagenet_mean_preprocess_batch(style_image.copy())
                style_image = utils.preprocess_batch(style_image)

                feature_style = vgg(style_v)
                gram_style = [net.gram_matrix(y) for y in feature_style]

                xc = utils.subtract_imagenet_mean_preprocess_batch(x.copy())
                f_xc_c = vgg(xc)[1]

                style_model.set_target(style_image)
                y = style_model(x)
                y = utils.subtract_imagenet_mean_batch(y)
                features_y = vgg(y)

                # TODO: why the coefficient 2?
                content_loss = 2 * args.content_weight * mse_loss(features_y[1], f_xc_c)

                style_loss = 0.0
                for m in range(len(features_y)):
                    gram_y = net.gram_matrix(features_y[m])
                    _, C, _ = gram_style[m].shape
                    gram_s = tf.expand_dims(gram_style[m], 0).broadcast_to(args.batch_size, 1, C, C)
                    style_loss += 2 * args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])
                total_loss = content_loss + style_loss
                agg_content_loss += content_loss[0]
                agg_style_loss += style_loss[0]
            gradients = tape.gradient(total_loss, style_model.variables)
            optimizer.apply_gradients(zip(gradients, style_model.trainable_variables))

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.3f}\tstyle: {:.3f}\ttotal: {:.3f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if (batch_id + 1) % (4 * args.log_interval) == 0:
                # save model
                save_model_filename = "Epoch_" + str(e) + "iters_" + \
                    str(count) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
                    args.content_weight) + "_" + str(args.style_weight) + ".params"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                tf.saved_model.save(style_model, save_model_path)
                print("\nCheckpoint, trained model saved at", save_model_path)

