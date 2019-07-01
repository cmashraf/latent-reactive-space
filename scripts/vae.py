#data
import pandas as pd
import numpy as np
from numpy.linalg import norm
import json
import random
import copy

#ML
import keras
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
from keras import objectives
from keras.objectives import binary_crossentropy #objs or losses
from keras.layers import Dense, Dropout, Input, Multiply, Add, Lambda, concatenate
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D


class MoleculeVAE():

    autoencoder = None
    
    def create(self,
               charset,
               max_length = 62,
               latent_rep_size = 292,
               weights_file = None,
               qspr = False,
               mol_inputs=1,
               conv_layers=3,
               gru_layers=3,
               conv_epsilon_std=0.01,
               conv_filters=3, 
               conv_kernel_size=9,
               gru_output_units=501):
        
        charset_length = len(charset)
        
        if mol_inputs == 1:
            x = Input(shape=(max_length, charset_length), name='one_hot_input_encoder')
        elif mol_inputs ==2:
            x1 = Input(shape=(max_length, charset_length), name='one_hot_cation_input')
            x2 = Input(shape=(max_length, charset_length), name='one_hot_anion_input')
            x = [x1, x2]
        
        ###ENCODER/DECODER
        _, z = self._buildEncoder(x, 
                                  latent_rep_size, 
                                  max_length,
                                  conv_epsilon_std,
                                  conv_layers, 
                                  conv_filters, 
                                  conv_kernel_size,
                                  mol_inputs)
        self.encoder = Model(x, z)
        encoded_input = Input(shape=(latent_rep_size,), name='encoded_input')
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length,
                gru_layers,
                gru_output_units,
                mol_inputs
            )
        )

        ###AUTOENCODER
        vae_loss, z1 = self._buildEncoder(x, 
                                          latent_rep_size, 
                                          max_length,
                                          conv_epsilon_std,
                                          conv_layers, 
                                          conv_filters,
                                          conv_kernel_size,
                                          mol_inputs)
        
        if qspr:
            self.autoencoder = Model(
                x,
                self._buildDecoderQSPR(
                    z1,
                    latent_rep_size,
                    max_length,
                    charset_length,
                    gru_layers,
                    gru_output_units,
                    mol_inputs
                )
            )

        else:
            self.autoencoder = Model(
                x,
                self._buildDecoder(
                    z1,
                    latent_rep_size,
                    max_length,
                    charset_length,
                    gru_layers,
                    gru_output_units,
                    mol_inputs
                )
            )
            
        self.qspr = Model(
            x,
            self._buildQSPR(
                z1,
                latent_rep_size,
                max_length,
                charset_length
            )
        )


        if weights_file:
            self.autoencoder.load_weights(weights_file, by_name = True)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)
            self.qspr.load_weights(weights_file, by_name = True)
        if qspr:
            if mol_inputs == 1:
                self.autoencoder.compile(optimizer = 'Adam',
                                         loss = {'decoded_mean': vae_loss, 
                                                 'qspr': 'mean_squared_error'},
                                         metrics = ['accuracy', 'mse'])
            elif mol_inputs == 2:
                self.autoencoder.compile(optimizer = 'Adam',
                                         loss = {'decoded_cation_mean': vae_loss, 
                                                 'decoded_anion_mean': vae_loss, 
                                                 'qspr': 'mean_squared_error'},
                                         metrics = ['accuracy', 'mse'])
        else:
            if mol_inputs == 1:
                self.autoencoder.compile(optimizer = 'Adam',
                                         loss = {'decoded_mean': vae_loss},
                                         metrics = ['accuracy'])
            elif mol_inputs == 2:
                self.autoencoder.compile(optimizer = 'Adam',
                                         loss = {'decoded_cation_mean': vae_loss,
                                                 'decoded_anion_mean': vae_loss},
                                         metrics = ['accuracy'])
    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std = 0.01, conv_layers=3, 
                      conv_filters=9, conv_kernel_size=9, mol_inputs=1):
        if mol_inputs == 2:
            x = concatenate(x)
        h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
        for convolution in range(conv_layers-2):
            h = Convolution1D(conv_filters, conv_kernel_size, activation = 'relu', name='conv_{}'.
                              format(convolution+2))(h)
        h = Convolution1D(10, 11, activation = 'relu', name='conv_{}'.format(conv_layers))(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation = 'relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def _buildDecoderQSPR(self, z, latent_rep_size, max_length, charset_length, 
                          gru_layers=3, gru_output_units=501, mol_inputs=1):

        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='gru_1')(h)
        for gru in range(gru_layers-2):
            h = GRU(gru_output_units, return_sequences = True, name='gru_{}'.format(gru+2))(h)
        h = GRU(501, return_sequences = True, name='gru_{}'.format(gru_layers))(h)
        if mol_inputs == 1:
            smiles_decoded = TimeDistributed(Dense(charset_length, activation='softmax'), 
                                             name='decoded_mean')(h)
        elif mol_inputs == 2:
            cation_decoded = TimeDistributed(Dense(charset_length, activation='softmax'), 
                                             name='decoded_cation_mean')(h)
            anion_decoded = TimeDistributed(Dense(charset_length, activation='softmax'), 
                                            name='decoded_anion_mean')(h)

        h = Dense(latent_rep_size, name='qspr_input', activation='relu')(z)
        h = Dense(100, activation='relu', name='hl_1')(h)
        h = Dropout(0.5)(h)
        smiles_qspr = Dense(1, activation='linear', name='qspr')(h)
        
        if mol_inputs == 1:
            return smiles_decoded, smiles_qspr
        elif mol_inputs == 2:
            return cation_decoded, anion_decoded, smiles_qspr

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length, 
                      gru_layers=3, gru_output_units=501, mol_inputs=1):

        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='gru_1')(h)
        for gru in range(gru_layers-2):
            h = GRU(gru_output_units, return_sequences = True, name='gru_{}'.format(gru+2))(h)
        h = GRU(501, return_sequences = True, name='gru_{}'.format(gru_layers))(h)
        if mol_inputs == 1:
            smiles_decoded = TimeDistributed(Dense(charset_length, activation='softmax'), 
                                             name='decoded_mean')(h)
        elif mol_inputs ==2:
            cation_decoded = TimeDistributed(Dense(charset_length, activation='softmax'), 
                                             name='decoded_cation_mean')(h)
            anion_decoded = TimeDistributed(Dense(charset_length, activation='softmax'), 
                                            name='decoded_anion_mean')(h)

        if mol_inputs == 1:
            return smiles_decoded
        elif mol_inputs == 2:
            return cation_decoded, anion_decoded
    
    def _buildQSPR(self, z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input', activation='relu')(z)
        h = Dense(100, activation='relu', name='hl_1')(h)
        h = Dropout(0.5)(h)
        return Dense(1, activation='linear', name='qspr')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, latent_rep_size = 292):
        self.create(charset, weights_file = weights_file, latent_rep_size = latent_rep_size)

        
class TwoMoleculeVAE():

    autoencoder = None
    
    def create(self,
               cat_charset,
               ani_charset,
               cat_max_length = 62,
               ani_max_length = 62,
               latent_rep_size = 292,
               weights_file = None,
               qspr = False,
               qspr_outputs = 1):
        
        ### cation
        max_length = cat_max_length
        charset = cat_charset
        charset_length = len(charset)
        x = Input(shape=(max_length, charset_length))
        _, z = self._buildEncoder('cation',
                                  x, 
                                  latent_rep_size,
                                  max_length)
        self.cation_encoder = Model(x, z)
            
        encoded_input = Input(shape=(latent_rep_size,))
        self.cation_decoder = Model(
            encoded_input,
            self._buildDecoder('cation',
                               encoded_input,
                               latent_rep_size,
                               max_length,
                               charset_length
            )
        )

        ### anion
        max_length = ani_max_length
        charset = ani_charset
        charset_length = len(ani_charset)
        x = Input(shape=(max_length, charset_length))
        _, z = self._buildEncoder('anion',
                                  x, 
                                  latent_rep_size,
                                  max_length)
        self.anion_encoder = Model(x, z)
            
        encoded_input = Input(shape=(latent_rep_size,))
        self.anion_decoder = Model(
            encoded_input,
            self._buildDecoder('anion',
                               encoded_input,
                               latent_rep_size,
                               max_length,
                               charset_length
            )
        )
        ###AUTOENCODER
        max_length = cat_max_length
        charset = cat_charset
        charset_length = len(charset)
        cat_x1 = Input(shape=(max_length, charset_length))
        cat_vae_loss, cat_z1 = self._buildEncoder('cation', cat_x1, latent_rep_size, max_length)
        
        max_length = ani_max_length
        charset = ani_charset
        charset_length = len(ani_charset)
        ani_x1 = Input(shape=(max_length, charset_length))
        ani_vae_loss, ani_z1 = self._buildEncoder('anion', ani_x1, latent_rep_size, max_length)
        
        
        if qspr:
            self.autoencoder = Model(
                [cat_x1, ani_x1],
                self._buildAutoencoderQSPR(
                    cat_z1,
                    ani_z1,
                    latent_rep_size,
                    cat_max_length,
                    len(cat_charset),
                    ani_max_length,
                    len(ani_charset),
                    qspr_outputs
                )
            )
        else:
            self.autoencoder = Model(
                [cat_x1, ani_x1],
                self._buildAutoencoder(
                    cat_z1,
                    ani_z1,
                    latent_rep_size,
                    cat_max_length,
                    len(cat_charset),
                    ani_max_length,
                    len(ani_charset)
                )
            )
        
        ###QSPR
        self.qspr = Model(
            [cat_x1, ani_x1],
            self._buildQSPR(
                cat_z1,
                ani_z1,
                latent_rep_size,
                max_length,
                charset_length,
                qspr_outputs
            )
        )
        if weights_file:
            self.cation_encoder.load_weights(weights_file, by_name = True)
            self.cation_decoder.load_weights(weights_file, by_name = True)
            self.anion_encoder.load_weights(weights_file, by_name = True)
            self.anion_decoder.load_weights(weights_file, by_name = True)
            self.autoencoder.load_weights(weights_file, by_name = True)
            self.qspr.load_weights(weights_file, by_name = True)
        if qspr:
            self.autoencoder.compile(optimizer = 'Adam',
                                     loss = {'cation_decoded_mean': cat_vae_loss,
                                             'anion_decoded_mean': ani_vae_loss,
                                             'qspr': 'mean_squared_error'},
                                     metrics = ['accuracy', 'mse'])
        else:
            self.autoencoder.compile(optimizer = 'Adam',
                                     loss = {'cation_decoded_mean': cat_vae_loss,
                                             'anion_decoded_mean': ani_vae_loss},
                                     metrics = ['accuracy'])
            
            
    def _buildEncoder(self, my_name, x, latent_rep_size, max_length, epsilon_std = 0.01):
        h = Convolution1D(9, 9, activation = 'relu', name='{}_conv_1'.format(my_name))(x)
        h = Convolution1D(9, 9, activation = 'relu', name='{}_conv_2'.format(my_name))(h)
        h = Convolution1D(10, 11, activation = 'relu', name='{}_conv_3'.format(my_name))(h)
        h = Flatten(name='{}_flatten_1'.format(my_name))(h)
        h = Dense(435, activation = 'relu', name='{}_dense_1'.format(my_name))(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), 
                                      mean=0., stddev = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='{}_z_mean'.format(my_name), 
                       activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='{}_z_log_var'.format(my_name), 
                          activation = 'linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), 
                                 name='{}_lambda'.format(my_name))([z_mean, z_log_var]))

    def _buildAutoencoderQSPR(self, cat_z, ani_z, latent_rep_size, 
                          cat_max_length, cat_charset_length,
                          ani_max_length, ani_charset_length,
                          qspr_outputs):

        h = Dense(latent_rep_size, name='cation_latent_input', activation = 'relu')(cat_z)
        h = RepeatVector(cat_max_length, name='cation_repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='cation_gru_1')(h)
        h = GRU(501, return_sequences = True, name='cation_gru_2')(h)
        h = GRU(501, return_sequences = True, name='cation_gru_3')(h)
        cat_smiles_decoded = TimeDistributed(Dense(cat_charset_length, 
                                                   activation='softmax'), 
                                             name='cation_decoded_mean')(h)
        
        h = Dense(latent_rep_size, name='anion_latent_input', activation = 'relu')(ani_z)
        h = RepeatVector(ani_max_length, name='anion_repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='anion_gru_1')(h)
        h = GRU(501, return_sequences = True, name='anion_gru_2')(h)
        h = GRU(501, return_sequences = True, name='anion_gru_3')(h)
        ani_smiles_decoded = TimeDistributed(Dense(ani_charset_length, 
                                                   activation='softmax'), 
                                             name='anion_decoded_mean')(h)
        
        combined = concatenate([cat_z, ani_z])
        h = Dense(latent_rep_size*2, name='qspr_input', activation='relu')(combined)
        h = Dense(100, activation='relu', name='hl_1')(h)
        h = Dropout(0.5)(h)
        
        smiles_qspr = Dense(qspr_outputs, activation='linear', name='qspr')(h)

        return cat_smiles_decoded, ani_smiles_decoded, smiles_qspr

    def _buildAutoencoder(self, cat_z, ani_z, latent_rep_size, 
                          cat_max_length, cat_charset_length,
                          ani_max_length, ani_charset_length):

        h = Dense(latent_rep_size, name='cation_latent_input', activation = 'relu')(cat_z)
        h = RepeatVector(cat_max_length, name='cation_repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='cation_gru_1')(h)
        h = GRU(501, return_sequences = True, name='cation_gru_2')(h)
        h = GRU(501, return_sequences = True, name='cation_gru_3')(h)
        cat_smiles_decoded = TimeDistributed(Dense(cat_charset_length, 
                                                   activation='softmax'), 
                                             name='cation_decoded_mean')(h)
        
        h = Dense(latent_rep_size, name='anion_latent_input', activation = 'relu')(ani_z)
        h = RepeatVector(ani_max_length, name='anion_repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='anion_gru_1')(h)
        h = GRU(501, return_sequences = True, name='anion_gru_2')(h)
        h = GRU(501, return_sequences = True, name='anion_gru_3')(h)
        ani_smiles_decoded = TimeDistributed(Dense(ani_charset_length, 
                                                   activation='softmax'), 
                                             name='anion_decoded_mean')(h)

        return cat_smiles_decoded, ani_smiles_decoded
    
    def _buildDecoder(self, my_name, z, latent_rep_size, max_length, charset_length):

        h = Dense(latent_rep_size, name='{}_latent_input'.format(my_name), activation = 'relu')(z)
        h = RepeatVector(max_length, name='{}_repeat_vector'.format(my_name))(h)
        h = GRU(501, return_sequences = True, name='{}_gru_1'.format(my_name))(h)
        h = GRU(501, return_sequences = True, name='{}_gru_2'.format(my_name))(h)
        h = GRU(501, return_sequences = True, name='{}_gru_3'.format(my_name))(h)
        smiles_decoded = TimeDistributed(Dense(charset_length, activation='softmax'),
                                         name='{}_decoded_mean'.format(my_name))(h)

        return smiles_decoded
    
    def _buildQSPR(self, cat_z, ani_z, latent_rep_size, max_length, charset_length, qspr_outputs):
        combined = concatenate([cat_z, ani_z])
        h = Dense(latent_rep_size*2, name='qspr_input', activation='relu')(combined)
        h = Dense(100, activation='relu', name='hl_1')(h)
        h = Dropout(0.5)(h)
        return Dense(qspr_outputs, activation='linear', name='qspr')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, latent_rep_size = 292):
        self.create(charset, weights_file = weights_file, latent_rep_size = latent_rep_size)
        

class TwoMoleculeOneLatentVAE():
    
    autoencoder = None
    
    def create(self,
               charset,
               max_length = 62,
               latent_rep_size = 292,
               weights_file = None,
               qspr = False):
        charset_length = len(charset)
        
        
        ###ENCODER/DECODER
        x1 = Input(shape=(max_length, charset_length), name='one_hot_cation_input')
        x2 = Input(shape=(max_length, charset_length), name='one_hot_anion_input')
        _, z = self._buildEncoder(x1,
                                  x2, 
                                  latent_rep_size,
                                  max_length)
        self.encoder = Model([x1, x2], z)
            
        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = Model(
            encoded_input,
            self._buildAutoencoder(encoded_input,
                               latent_rep_size,
                               max_length,
                               charset_length
            )
        )


        ###AUTOENCODER
        charset_length = len(charset)
        cat_x1 = Input(shape=(max_length, charset_length), name='one_hot_cation_input')
        ani_x1 = Input(shape=(max_length, charset_length), name='one_hot_anion_input')
        vae_loss, z1 = self._buildEncoder(cat_x1, ani_x1, latent_rep_size, max_length)
        
        
        
        if qspr:
            self.autoencoder = Model(
                [cat_x1, ani_x1],
                self._buildAutoencoderQSPR(
                    z1,
                    latent_rep_size,
                    max_length,
                    len(charset)
                )
            )
        else:
            self.autoencoder = Model(
                [cat_x1, ani_x1],
                self._buildAutoencoder(
                    z1,
                    latent_rep_size,
                    max_length,
                    len(charset)
                )
            )
        
        ###QSPR
        self.qspr = Model(
            [cat_x1, ani_x1],
            self._buildQSPR(
                z1,
                latent_rep_size,
                max_length,
                charset_length
            )
        )
        if weights_file:
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)
            self.autoencoder.load_weights(weights_file, by_name = True)
#             self.qspr.load_weights(weights_file, by_name = True)
        if qspr:
            self.autoencoder.compile(optimizer = 'Adam',
                                     loss = {'cation_decoded_mean': vae_loss,
                                             'anion_decoded_mean': vae_loss,
                                             'qspr': 'mean_squared_error'},
                                     metrics = ['accuracy', 'mse'])
        else:
            self.autoencoder.compile(optimizer = 'Adam',
                                     loss = {'cation_decoded_mean': vae_loss,
                                             'anion_decoded_mean': vae_loss},
                                     metrics = ['accuracy'])
            
            
    def _buildEncoder(self, x1, x2, latent_rep_size, max_length, epsilon_std = 0.01):
        h = Convolution1D(9, 9, activation = 'relu', name='conv_1a')(x1)
        h = Convolution1D(9, 9, activation = 'relu', name='conv_2a')(h)
        h = Convolution1D(10, 11, activation = 'relu', name='conv_3a')(h)
        
        h2 = Convolution1D(9, 9, activation = 'relu', name='conv_1b')(x2)
        h2 = Convolution1D(9, 9, activation = 'relu', name='conv_2b')(h2)
        h2 = Convolution1D(10, 11, activation = 'relu', name='conv_3b')(h2)
        
        combined = concatenate([h, h2])
        h3 = Convolution1D(9, 9, activation = 'relu', name='conv_1c')(combined)
        h3 = Convolution1D(9, 9, activation = 'relu', name='conv_2c')(h3)
        h3 = Convolution1D(10, 11, activation = 'relu', name='conv_3c')(h3)
        h3 = Flatten(name='flatten_1c')(h3)
        h3 = Dense(435, activation = 'relu', name='dense_1c')(h3)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), 
                                      mean=0., stddev = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', 
                       activation = 'linear')(h3)
        z_log_var = Dense(latent_rep_size, name='z_log_var', 
                          activation = 'linear')(h3)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), 
                                 name='lambda')([z_mean, z_log_var]))

    def _buildAutoencoderQSPR(self, z, latent_rep_size, 
                              max_length, charset_length):

        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='gru_1')(h)
        h = GRU(501, return_sequences = True, name='gru_2')(h)
        h = GRU(501, return_sequences = True, name='gru_3')(h)
        
        h2 = GRU(501, return_sequences = True, name='cation_gru_1')(h)
        h2 = GRU(501, return_sequences = True, name='cation_gru_2')(h2)
        h2 = GRU(501, return_sequences = True, name='cation_gru_3')(h2)
        cat_smiles_decoded = TimeDistributed(Dense(charset_length, 
                                                   activation='softmax'), 
                                             name='cation_decoded_mean')(h2)
        
        h3 = GRU(501, return_sequences = True, name='anion_gru_1')(h)
        h3 = GRU(501, return_sequences = True, name='anion_gru_2')(h3)
        h3 = GRU(501, return_sequences = True, name='anion_gru_3')(h3)
        ani_smiles_decoded = TimeDistributed(Dense(charset_length, 
                                                   activation='softmax'), 
                                             name='anion_decoded_mean')(h3)
        
        h = Dense(latent_rep_size*2, name='qspr_input', activation='relu')(z)
        h = Dense(100, activation='relu', name='hl_1')(h)
        h = Dropout(0.5)(h)
        smiles_qspr = Dense(1, activation='linear', name='qspr')(h)

        return cat_smiles_decoded, ani_smiles_decoded, smiles_qspr

    def _buildAutoencoder(self, z, latent_rep_size, 
                          max_length, charset_length):

        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='gru_1')(h)
        h = GRU(501, return_sequences = True, name='gru_2')(h)
        h = GRU(501, return_sequences = True, name='gru_3')(h)
        
        h2 = GRU(501, return_sequences = True, name='cation_gru_1')(h)
        h2 = GRU(501, return_sequences = True, name='cation_gru_2')(h2)
        h2 = GRU(501, return_sequences = True, name='cation_gru_3')(h2)
        cat_smiles_decoded = TimeDistributed(Dense(charset_length, 
                                                   activation='softmax'), 
                                             name='cation_decoded_mean')(h2)
        
        h3 = GRU(501, return_sequences = True, name='anion_gru_1')(h)
        h3 = GRU(501, return_sequences = True, name='anion_gru_2')(h3)
        h3 = GRU(501, return_sequences = True, name='anion_gru_3')(h3)
        ani_smiles_decoded = TimeDistributed(Dense(charset_length, 
                                                   activation='softmax'), 
                                             name='anion_decoded_mean')(h3)

        return cat_smiles_decoded, ani_smiles_decoded
    
    def _buildQSPR(self, z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size*2, name='qspr_input', activation='relu')(z)
        h = Dense(100, activation='relu', name='hl_1')(h)
        h = Dropout(0.5)(h)
        return Dense(1, activation='linear', name='qspr')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, latent_rep_size = 292):
        self.create(charset, weights_file = weights_file, latent_rep_size = latent_rep_size)