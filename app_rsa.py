
#from keras.preprocessing import image
#import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np
import os
import random
import scipy.misc
from tqdm import *






from flask import *  
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw

def rsaencrypt(message):
    import rsa
  
    publicKey, privateKey = rsa.newkeys(512)

    encMessage = rsa.encrypt(message.encode(), publicKey)

    decMessage = rsa.decrypt(encMessage, privateKey).decode()
    print(decMessage)
    
    return encMessage

app = Flask(__name__)  
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app_root = os.path.dirname(os.path.abspath("__file__"))
@app.route('/upload',methods=['GET', 'POST'])  
def upload():  
    import pathlib
    try:
        path = pathlib.Path('static/cover/cover_image.png')
        path.unlink()
    except:
        print("")
    return render_template("Page.html")  
 
@app.route('/success', methods = ['GET','POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save('static/cover/'+f.filename)
        os.rename('static/cover/'+f.filename, 'static/cover/cover_image.png')
    return render_template("success.html")  
    
@app.route('/secret', methods = ['POST'])  
def secret():  
    if request.method == 'POST':  
        first_name = request.form.get("fname")
        name = rsaencrypt(first_name)
        img = Image.new('RGB', (64,64), color = (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((10,10), str(name), fill=(0,0,0))
        img.save('static/secret/secret_image.png')
        # Variable used to weight the losses of the secret and cover images (See paper for more details)
        beta = 1.0
            
        # Loss for reveal network
        def rev_loss(s_true, s_pred):
            # Loss for reveal network is: beta * |S-S'|
            return beta * K.sum(K.square(s_true - s_pred))
        
        # Loss for the full model, used for preparation and hidding networks
        def full_loss(y_true, y_pred):
            # Loss for the full model is: |C-C'| + beta * |S-S'|
            s_true, c_true = y_true[:,:,:,0:3], y_true[:,:,:,3:6]
            s_pred, c_pred = y_pred[:,:,:,0:3], y_pred[:,:,:,3:6]
            
            s_loss = beta * K.sum(K.square(s_true - s_pred))
            c_loss = K.sum(K.square(c_true - c_pred))
            
            return s_loss + c_loss
        
        
        # Returns the encoder as a Keras model, composed by Preparation and Hiding Networks.
        def make_encoder(input_size):
            input_S = Input(shape=(input_size))
            input_C= Input(shape=(input_size))
        
            # Preparation Network
            x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_3x3')(input_S)
            x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_4x4')(input_S)
            x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_5x5')(input_S)
            x = concatenate([x3, x4, x5])
            
            x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_3x3')(x)
            x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_4x4')(x)
            x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_5x5')(x)
            x = concatenate([x3, x4, x5])
            
            x = concatenate([input_C, x])
            
            # Hiding network
            x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid0_3x3')(x)
            x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid0_4x4')(x)
            x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid0_5x5')(x)
            x = concatenate([x3, x4, x5])
            
            x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid1_3x3')(x)
            x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid1_4x4')(x)
            x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid1_5x5')(x)
            x = concatenate([x3, x4, x5])
            
            x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid2_3x3')(x)
            x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid2_4x4')(x)
            x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid2_5x5')(x)
            x = concatenate([x3, x4, x5])
            
            x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid3_3x3')(x)
            x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid3_4x4')(x)
            x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid3_5x5')(x)
            x = concatenate([x3, x4, x5])
            
            x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid4_3x3')(x)
            x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid4_4x4')(x)
            x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid5_5x5')(x)
            x = concatenate([x3, x4, x5])
            
            output_Cprime = Conv2D(3, (3, 3), strides = (1, 1), padding='same', activation='relu', name='output_C')(x)
            
            return Model(inputs=[input_S, input_C],
                         outputs=output_Cprime,
                         name = 'Encoder')
        
        # Returns the decoder as a Keras model, composed by the Reveal Network
        def make_decoder(input_size, fixed=False):
            
            # Reveal network
            reveal_input = Input(shape=(input_size))
            
            # Adding Gaussian noise with 0.01 standard deviation.
            input_with_noise = GaussianNoise(0.01, name='output_C_noise')(reveal_input)
            
            x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_3x3')(input_with_noise)
            x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_4x4')(input_with_noise)
            x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_5x5')(input_with_noise)
            x = concatenate([x3, x4, x5])
            
            x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_3x3')(x)
            x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_4x4')(x)
            x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_5x5')(x)
            x = concatenate([x3, x4, x5])
            
            x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_3x3')(x)
            x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_4x4')(x)
            x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_5x5')(x)
            x = concatenate([x3, x4, x5])
            
            x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_3x3')(x)
            x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_4x4')(x)
            x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_5x5')(x)
            x = concatenate([x3, x4, x5])
            
            x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev4_3x3')(x)
            x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev4_4x4')(x)
            x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev5_5x5')(x)
            x = concatenate([x3, x4, x5])
            
            output_Sprime = Conv2D(3, (3, 3), strides = (1, 1), padding='same', activation='relu', name='output_S')(x)
            
            if not fixed:
                return Model(inputs=reveal_input,
                             outputs=output_Sprime,
                             name = 'Decoder')
            else:
                return Container(inputs=reveal_input,
                                 outputs=output_Sprime,
                                 name = 'DecoderFixed')
        
        # Full model.
        def make_model(input_size):
            input_S = Input(shape=(input_size))
            input_C= Input(shape=(input_size))
            
            encoder = make_encoder(input_size)
            
            decoder = make_decoder(input_size)
            decoder.compile(optimizer='adam', loss=rev_loss)
            decoder.trainable = False
            
            output_Cprime = encoder([input_S, input_C])
            output_Sprime = decoder(output_Cprime)
        
            autoencoder = Model(inputs=[input_S, input_C],
                                outputs=concatenate([output_Sprime, output_Cprime]))
            autoencoder.compile(optimizer='adam', loss=full_loss)
            
            return encoder, decoder, autoencoder
        
        encoder_model, reveal_model, autoencoder_model = make_model([64,64])
        
        def loadCover():
            X_train=[]
         
            for c in os.listdir("static/cover"):
                    img_i = image.load_img(os.path.join("static/cover", c))
                    img_i = img_i.resize((64,64))
                    x = image.img_to_array(img_i)
                    X_train.append(x)
            return np.array(X_train,dtype=float)
    
        def loadSecret():
            X_train=[]
            for c in os.listdir("static/secret"):
                    img_i = image.load_img(os.path.join("static/secret", c))
                    img_i = img_i.resize((64,64))
                    x = image.img_to_array(img_i)
                    X_train.append(x)
            return np.array(X_train,dtype=float)

    
        X_train_orig=loadCover()
        X_test_orig=loadSecret()
        
        X_train = X_train_orig/255.
        X_test = X_test_orig/255.
        print ("Number of cover examples = " + str(X_train.shape[0]))
        print ("X_train shape: " + str(X_train.shape)) # Should be (train_size, 64, 64, 3).
        print ("Number of secret examples = " + str(X_test.shape[0]))
        print ("X_train shape: " + str(X_test.shape)) # Should be (train_size, 64, 64, 3).
        
        
        input_S = X_test
        input_C = X_train
        autoencoder_model.load_weights(r'C:\Users\Thamil Vani\Downloads\singlemodel500.hdf5')
        # Retrieve decoded predictions.
        decoded = autoencoder_model.predict([input_S, input_C])
        decoded_S, decoded_C = decoded[...,0:3], decoded[...,3:6]
        
        # Get absolute difference between the outputs and the expected values.
        diff_S, diff_C = np.abs(decoded_S - input_S), np.abs(decoded_C - input_C) 
        
        def pixel_errors(input_S, input_C, decoded_S, decoded_C):
            """Calculates mean of Sum of Squared Errors per pixel for cover and secret images. """
            see_Spixel = np.sqrt(np.mean(np.square(255*(input_S - decoded_S))))
            see_Cpixel = np.sqrt(np.mean(np.square(255*(input_C - decoded_C))))
            
            return see_Spixel, see_Cpixel
        
        def pixel_histogram(diff_S, diff_C):
            """Calculates histograms of errors for cover and secret image. """
            diff_Sflat = diff_S.flatten()
            diff_Cflat = diff_C.flatten()
            
            fig = plt.figure(figsize=(15, 5))
            a=fig.add_subplot(1,2,1)
                
            imgplot = plt.hist(255* diff_Cflat, 100, density=1, alpha=0.75, facecolor='red')
            a.set_title('Distribution of error in the Cover image.')
            plt.axis([0, 250, 0, 0.2])
            
            a=fig.add_subplot(1,2,2)
            imgplot = plt.hist(255* diff_Sflat, 100, density=1, alpha=0.75, facecolor='red')
            a.set_title('Distribution of errors in the Secret image.')
            plt.axis([0, 250, 0, 0.2])
            
            plt.show()
            
            # Print pixel-wise average errors in a 256 scale.
            S_error, C_error = pixel_errors(input_S, input_C, decoded_S, decoded_C)
            
            print ("S error per pixel [0, 255]:", S_error)
            print ("C error per pixel [0, 255]:", C_error)
            
            pixel_histogram(diff_S, diff_C)
            
                
            # Show images in gray scale
            SHOW_GRAY = False
            # Show difference bettwen predictions and ground truth.
            SHOW_DIFF = True
            
            # Diff enhance magnitude
            ENHANCE = 1
            
            # Number of secret and cover pairs to show.
            n = 6
            
            def rgb2gray(rgb):
                return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
            
            def show_image(img, n_rows, n_col, idx, gray=False, first_row=False, title=None):
                ax = plt.subplot(n_rows, n_col, idx)
                if gray:
                    plt.imshow(rgb2gray(img), cmap = plt.get_cmap('gray'))
                else:
                    plt.imshow(img)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if first_row:
                    plt.title(title)
            
            plt.figure(figsize=(14, 15))
            rand_indx = [random.randint(0, 1000) for x in range(n)]
            # for i, idx in enumerate(range(0, n)):
            for i, idx in enumerate(range(0, n)):
                n_col = 6 if SHOW_DIFF else 4
                
                show_image(input_C[idx], n, n_col, i * n_col + 1, gray=SHOW_GRAY, first_row=i==0, title='Cover')
            
                show_image(input_S[idx], n, n_col, i * n_col + 2, gray=SHOW_GRAY, first_row=i==0, title='Secret')
                
                show_image(decoded_C[idx], n, n_col, i * n_col + 3, gray=SHOW_GRAY, first_row=i==0, title='Encoded Cover')
                
                show_image(decoded_S[idx], n, n_col, i * n_col + 4, gray=SHOW_GRAY, first_row=i==0, title='Decoded Secret')
            
                
                if SHOW_DIFF:
                    show_image(np.multiply(diff_C[idx], ENHANCE), n, n_col, i * n_col + 5, gray=SHOW_GRAY, first_row=i==0, title='Diff Cover')
                    
                    show_image(np.multiply(diff_S[idx], ENHANCE), n, n_col, i * n_col + 6, gray=SHOW_GRAY, first_row=i==0, title='Diff Secret')
            
            plt.show()

    return render_template("secret.html")  
    


    
    
if __name__ == '__main__':  
    app.run(debug = False)  






