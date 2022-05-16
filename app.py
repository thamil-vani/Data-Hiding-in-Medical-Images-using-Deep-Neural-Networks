from keras.preprocessing import image
import keras.backend as K
#import matplotlib.pyplot as plt
import numpy as np
import os
import random
from blake3 import blake3
import scipy.misc
from tqdm import *

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.layers import *
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from PIL import ImageFont
from PIL import Image as im
from flask import *
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import zipfile
from tinyec import registry
from Crypto.Cipher import AES
import hashlib, secrets, binascii

import matplotlib
import hashlib
import binascii
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
encryptedMsg=""
privKey=""
def encrypt_AES_GCM(msg, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM)
    ciphertext, authTag = aesCipher.encrypt_and_digest(msg)
    return (ciphertext, aesCipher.nonce, authTag)

def decrypt_AES_GCM(ciphertext, nonce, authTag, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM, nonce)
    plaintext = aesCipher.decrypt_and_verify(ciphertext, authTag)
    return plaintext

def ecc_point_to_256_bit_key(point):
    sha = hashlib.sha256(int.to_bytes(point.x, 32, 'big'))
    sha.update(int.to_bytes(point.y, 32, 'big'))
    return sha.digest()

curve = registry.get_curve('brainpoolP256r1')

def encrypt_ECC(msg, pubKey):
    ciphertextPrivKey = secrets.randbelow(curve.field.n)
    sharedECCKey = ciphertextPrivKey * pubKey
    secretKey = ecc_point_to_256_bit_key(sharedECCKey)
    ciphertext, nonce, authTag = encrypt_AES_GCM(msg, secretKey)
    ciphertextPubKey = ciphertextPrivKey * curve.g
    return (ciphertext, nonce, authTag, ciphertextPubKey)

def decrypt_ECC(cipher_text_entered,encryptedMsg, privKey):
    (ciphertext, nonce, authTag, ciphertextPubKey) = encryptedMsg
    cp_text = binascii.hexlify(encryptedMsg[0]).decode("utf-8")
    if str(cp_text) != cipher_text_entered:
        print(cp_text,cipher_text_entered)
        return ""
    sharedECCKey = privKey * ciphertextPubKey
    secretKey = ecc_point_to_256_bit_key(sharedECCKey)
    plaintext = decrypt_AES_GCM(ciphertext, nonce, authTag, secretKey)
    return plaintext

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app_root = os.path.dirname(os.path.abspath("__file__"))
dataset=1




@app.route('/upload' ,methods=['GET', 'POST'])
def upload():
    import pathlib
    try:
        path = pathlib.Path('static/cover_image.png')
        path.unlink()
    except:
        print("")
    return render_template("Page.html")

@app.route('/success', methods = ['GET' ,'POST'])
def success():
    if request.method == 'POST':
        global dataset
        f = request.files['file']
        f.save('static/ ' +f.filename)

        if(f.filename.find("MRI")>0):
            dataset=1
        elif (f.filename.find("DRIVE") > 0):
            dataset =2
        os.rename('static/ ' +f.filename, 'static/cover_image.png')
    return render_template("success.html")

@app.route('/secret', methods = ['POST'])
def secret():
    if request.method == 'POST':
        global encryptedMsg
        global privKey
        global dataset
        secret_msg = request.form.get("fname")
        #key=request.form.get("key")
        #print(key)
        #secret_msg = onetimepad.encrypt(secret_msg, key)
        msg = bytes(secret_msg, 'utf-8')
        print("original msg:", str(msg))
        privKey = secrets.randbelow(curve.field.n)
        pubKey = privKey * curve.g
        encryptedMsg = encrypt_ECC(msg, pubKey)
        ciphertext,nonce_d,authTag_d,ciphertextPubKey_d = encryptedMsg
        print("encrypted msg:", encryptedMsg)
        print("encrypted msg:", binascii.hexlify(encryptedMsg[0]).decode("utf-8"))
        name=binascii.hexlify(encryptedMsg[0]).decode("utf-8")
        print("name",name)
        img = Image.new('RGB', (128 ,128), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        fnt = ImageFont.truetype(r"C:\Users\Thamil Vani\Downloads\verdana\verdanab.ttf", 15)
        #fnt1 = ImageFont.truetype(r"C:\Users\Thamil Vani\Downloads\arial\arial.ttf", 18)
        result=""
        count=0
        for char in name:
            if(count<8):
                result=result+char
                count+=1
            else:
                result=result+"\n"+char
                count=0
        d.text((20, 25), result, font=fnt,fill=(255, 255, 0))
        #d.text((20, 80), "\n"+key, font=fnt1, fill=(255, 255, 0))
        img.save('static/secret_image.png')
        beta = 1.0
        def rev_loss(s_true, s_pred):
            return beta * K.sum(K.square(s_true - s_pred))

        def full_loss(y_true, y_pred):

            s_true, c_true = y_true[: ,: ,: ,0:3], y_true[: ,: ,: ,3:6]
            s_pred, c_pred = y_pred[: ,: ,: ,0:3], y_pred[: ,: ,: ,3:6]

            s_loss = beta * K.sum(K.square(s_true - s_pred))
            c_loss = K.sum(K.square(c_true - c_pred))

            return s_loss + c_loss

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


        def make_decoder(input_size, fixed=False):
            # Reveal network
            reveal_input = Input(shape=(input_size))
            input_with_noise = GaussianNoise(0.01, name='output_C_noise')(reveal_input)

            x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_3x3') \
                (input_with_noise)
            x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_4x4') \
                (input_with_noise)
            x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_5x5') \
                (input_with_noise)
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

        encoder_model, reveal_model, autoencoder_model = make_model([128 ,128,3])

        def loadCover():
            X_train =[]
            img_i = image.load_img(os.path.join("static", "cover_image.png"))
            img_i = img_i.resize((128 ,128))
            x = image.img_to_array(img_i)
            X_train.append(x)
            return np.array(X_train ,dtype=float)

        def loadSecret():
            X_train =[]
            img_i = image.load_img(os.path.join("static", "secret_image.png"))
            img_i = img_i.resize((128, 128))
            x = image.img_to_array(img_i)
            X_train.append(x)
            return np.array(X_train ,dtype=float)

        X_train_orig =loadCover()
        X_test_orig =loadSecret()

        X_train = X_train_orig /255.
        X_test = X_test_orig /255.
        # print ("Number of cover examples = " + str(X_train.shape[0]))
        # print ("X_train shape: " + str(X_train.shape))
        # print ("Number of secret examples = " + str(X_test.shape[0]))
        # print ("X_train shape: " + str(X_test.shape))

        input_S = X_test
        input_C = X_train
        print("Dataset " ,dataset)
        if(dataset==1):
            autoencoder_model.load_weights(r'static/model/singlemodelind.hdf5')
        else:
            autoencoder_model.load_weights(r'static/model/singlemodeldrive.hdf5')


        decoded = autoencoder_model.predict([input_S, input_C])


        decoded_S, decoded_C = decoded[... ,0:3], decoded[... ,3:6]

        SHOW_GRAY = False
        SHOW_DIFF = False
        n = 1
        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

        def show_image_container(img, n_rows, n_col, idx, gray=False, first_row=False, title=None):
            if gray:
                plt.imshow(rgb2gray(img), cmap = plt.get_cap('gray'))
            else:
                plt.imshow(img)
                plt.axis('off')
                plt.savefig('static/container_image.png')

        def show_image_dsecret(img, n_rows, n_col, idx, gray=False, first_row=False, title=None):
            if gray:
                plt.imshow(rgb2gray(img), cmap = plt.get_cap('gray'))
            else:
                plt.imshow(img)
                plt.axis('off')
                plt.savefig('static/decoded_secret.png')

            #plt.figure(figsize=(14, 15))

        for i, idx in enumerate(range(0, n)):
            n_col = 1 if SHOW_DIFF else 1
            #print("Decoded cover : ",decoded_C[idx])
            data = Image.fromarray((decoded_C[idx] * 255).astype(np.uint8))
            data.save("static/container_image.png")
            # data = Image.fromarray((decoded_S[idx] * 255).astype(np.uint8))
            # data.save("static/decoded_secret.png")
            # show_image_container(decoded_C[idx], n, n_col, i * n_col + 1, gray=SHOW_GRAY, first_row=i==0, title= 'Encoded Cover')
            show_image_dsecret(decoded_S[idx], n, n_col, i * n_col + 1, gray=SHOW_GRAY, first_row=i==0, title= 'Decoded Secret')

    return render_template("secret.html")

@app.route('/download', methods = ['GET' ,'POST'])
def download():
    import hashlib
    def hash_file(filename):
        # h = hashlib.sha512()
        # with open(filename, 'rb') as file:
        #     chunk = 0
        #     while chunk != b'':
        #         chunk = file.read(1024)
        #         h.update(chunk)
        # return h.hexdigest()
        # import hashlib
        # Using hashlib.blake2b() method
        h = blake3()
        with open(filename, 'rb') as file:
            chunk = 0
            while chunk != b'':
                chunk = file.read(1024)
                h.update(chunk)
        return h.hexdigest()

    message = hash_file("static/container_image.png")
    print(message)

    with open('static/hashvalue.txt', 'w') as f:
            f.write(message)
    f.close()

    zipf = zipfile.ZipFile('files.zip', 'w', zipfile.ZIP_DEFLATED)
    zipf.write('static/' + "container_image.png")
    zipf.write('static/' + "hashvalue.txt")
    zipf.close()
    return send_file('files.zip',
                     mimetype='zip',
                     attachment_filename='files.zip',
                     as_attachment=True)
    #return send_file('static/container_image.png',attachment_filename='container_image.png',as_attachment=True)

@app.route('/receiver', methods = ['GET' ,'POST'])
def receiver():
    import pathlib
    try:
        path = pathlib.Path('static/cover_image_receiver.png')
        path.unlink()
    except:
        print("")
    return render_template("receiver.html")

@app.route('/hashing', methods = ['GET' ,'POST'])
def hashing():
    if request.method == 'POST':
        f = request.files['file']
        f.save('static/ ' + f.filename)
        os.rename('static/ ' + f.filename, 'static/cover_image_receiver.png')
    return render_template("hashing.html")

@app.route('/result', methods = ['GET' ,'POST'])
def result():
    if request.method == 'POST':
        import hashlib
        def hash_file(filename):
            h = blake3()
            with open(filename, 'rb') as file:
                chunk = 0
                while chunk != b'':
                    chunk = file.read(1024)
                    h.update(chunk)
            return h.hexdigest()

        hash_value_receiver = hash_file("static/cover_image_receiver.png")
        hash_value_sender = request.form.get("hash_value")
        integrity_text=""
        print("Sender")
        print(hash_value_sender)
        print("Receiver")
        print(hash_value_receiver)
        if(hash_value_sender==hash_value_receiver):
            integrity_text+="The images are same"
            return render_template("integrity1.html", text=integrity_text)
        else:
            integrity_text+="The images are not same"
            return render_template("integrity2.html", text=integrity_text)

@app.route('/decode', methods = ['GET' ,'POST'])
def decode():

    return render_template("result.html")
#receiver.hyml la change to hashing
@app.route('/decrypt', methods = ['GET' ,'POST'])
def decrypt():
    if request.method == 'POST':
        cipher_text_entered = request.form.get("enctext")
        #otp_key=request.form.get("otp")
        global privKey
        global encryptedMsg
        decryptedMsg = decrypt_ECC(cipher_text_entered,encryptedMsg, privKey)
        if decryptedMsg != "":
            decryptedMsg = decryptedMsg.decode("utf-8")
            #decryptedMsg = onetimepad.decrypt(decryptedMsg, otp_key)
        #print("decrypted msg:", decryptedMsg)
    return render_template("decrypt.html",text=decryptedMsg)

if __name__ == '__main__':
    app.run(debug = False)