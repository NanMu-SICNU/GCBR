import tensorflow as tf
import vgg16
import cv2
import numpy as np

img_size = 416
label_size = img_size / 2
fea_dim = 128

class Model:
    def __init__(self):
        self.vgg = vgg16.Vgg16()

        self.input_holder = tf.compat.v1.placeholder(tf.float32, [1, img_size, img_size, 3])
        self.label_holder = tf.compat.v1.placeholder(tf.float32, [label_size*label_size, 2])

        self.sobel_fx, self.sobel_fy = self.sobel_filter()

        self.contour_th = 1.5
        self.contour_weight = 0.0001

    def build_model(self):

        #build the VGG-16 model
        vgg = self.vgg
        vgg.build(self.input_holder)
       
       
        #Global Convolutional Network
        vgg.pool5 = self.GCN5(vgg.pool5)
        vgg.pool4 = self.GCN4(vgg.pool4)
        vgg.pool3 = self.GCN3(vgg.pool3)
        vgg.pool2 = self.GCN2(vgg.pool2)
        vgg.pool1 = self.GCN1(vgg.pool1)
        
        #Boundary Refinement
        vgg.pool5 = self.BR5(vgg.pool5)
        vgg.pool4 = self.BR4(vgg.pool4)
        vgg.pool3 = self.BR3(vgg.pool3)
        vgg.pool2 = self.BR2(vgg.pool2)
        vgg.pool1 = self.BR1(vgg.pool1)   
            
        #Global Feature and Global Score
        self.Fea_Global_1 = tf.nn.relu(self.Conv_2d(vgg.pool5, [7, 7, 512, fea_dim], 0.01,
                                                    padding='VALID', name='Fea_Global_1'))
        self.Fea_Global_1 = self.BR501(self.Fea_Global_1)
                                                     
        self.Fea_Global_2 = tf.nn.relu(self.Conv_2d(self.Fea_Global_1, [5, 5, fea_dim, fea_dim], 0.01,
                                                    padding='VALID', name='Fea_Global_2'))
        self.Fea_Global_2 = self.BR502(self.Fea_Global_2)                                            
                                                   
        self.Fea_Global = self.Conv_2d(self.Fea_Global_2, [3, 3, fea_dim, fea_dim], 0.01,
                                       padding='VALID', name='Fea_Global')
        self.Fea_Global = self.BR503(self.Fea_Global)                                
                                       
        
        #Local Score
        self.Fea_P5 = tf.nn.relu(self.Conv_2d(vgg.pool5, [3, 3, 512, fea_dim], 0.01, padding='SAME', name='Fea_P5'))
        self.Fea_P4 = tf.nn.relu(self.Conv_2d(vgg.pool4, [3, 3, 512, fea_dim], 0.01, padding='SAME', name='Fea_P4'))
        self.Fea_P3 = tf.nn.relu(self.Conv_2d(vgg.pool3, [3, 3, 256, fea_dim], 0.01, padding='SAME', name='Fea_P3'))
        self.Fea_P2 = tf.nn.relu(self.Conv_2d(vgg.pool2, [3, 3, 128, fea_dim], 0.01, padding='SAME', name='Fea_P2'))
        self.Fea_P1 = tf.nn.relu(self.Conv_2d(vgg.pool1, [3, 3, 64, fea_dim], 0.01, padding='SAME', name='Fea_P1'))
        
        #Boundary Refinement
        self.Fea_P5 = self.BR51(self.Fea_P5)
        self.Fea_P4 = self.BR41(self.Fea_P4)
        self.Fea_P3 = self.BR31(self.Fea_P3)
        self.Fea_P2 = self.BR21(self.Fea_P2)
        self.Fea_P1 = self.BR11(self.Fea_P1)

        self.Fea_P5_LC = self.Contrast_Layer(self.Fea_P5, 3)
        self.Fea_P4_LC = self.Contrast_Layer(self.Fea_P4, 3)
        self.Fea_P3_LC = self.Contrast_Layer(self.Fea_P3, 3)
        self.Fea_P2_LC = self.Contrast_Layer(self.Fea_P2, 3)
        self.Fea_P1_LC = self.Contrast_Layer(self.Fea_P1, 3)
        
        print('111')
        #Deconv Layer
        self.Fea_P5_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P5, self.Fea_P5_LC], axis=3),
                                                   [1, 26, 26, fea_dim], 5, 2, name='Fea_P5_Deconv'))
        print(self.Fea_P5_Up.shape)
        self.Fea_P5_Up = self.BR52(self.Fea_P5_Up)  
                                               
        self.Fea_P4_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P4, self.Fea_P4_LC, self.Fea_P5_Up], axis=3),
                                                   [1, 52, 52, fea_dim*2], 5, 2, name='Fea_P4_Deconv'))
        self.Fea_P4_Up = self.BR42(self.Fea_P4_Up)
                                                           
        self.Fea_P3_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P3, self.Fea_P3_LC, self.Fea_P4_Up], axis=3),
                                                   [1, 104, 104, fea_dim*3], 5, 2, name='Fea_P3_Deconv'))
        self.Fea_P3_Up = self.BR32(self.Fea_P3_Up)
                                                          
        self.Fea_P2_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P2, self.Fea_P2_LC, self.Fea_P3_Up], axis=3),
                                                   [1, 208,208 , fea_dim*4], 5, 2, name='Fea_P2_Deconv'))
        self.Fea_P2_Up = self.BR22(self.Fea_P2_Up)


        print('222') 
        self.Local_Fea = self.Conv_2d(tf.concat([self.Fea_P1, self.Fea_P1_LC, self.Fea_P2_Up], axis=3),
                                      [1, 1, fea_dim*6, fea_dim*5], 0.01, padding='VALID', name='Local_Fea')
         
        self.Local_Fea = self.BRLF(self.Local_Fea)
                               
        self.Local_Score = self.Conv_2d(self.Local_Fea, [1, 1, fea_dim*5, 2], 0.01, padding='VALID', name='Local_Score')
        self.Local_Score = self.BRLS(self.Local_Score)

        self.Global_Score = self.Conv_2d(self.Fea_Global,
                                         [1, 1, fea_dim, 2], 0.01, padding='VALID', name='Global_Score')
        self.Global_Score = self.BRGS(self.Global_Score)
        
        
        print(self.Local_Score.shape)
        print(self.Global_Score.shape)
        self.Score = self.Local_Score + self.Global_Score
        self.Score = tf.reshape(self.Score, [-1,2])

        self.Prob = tf.nn.softmax(self.Score)

        #Get the contour term
        self.Prob_C = tf.reshape(self.Prob, [1, 208, 208, 2])
        self.Prob_Grad = tf.tanh(self.im_gradient(self.Prob_C))
        self.Prob_Grad = tf.tanh(tf.reduce_sum(self.im_gradient(self.Prob_C), reduction_indices=3, keepdims=True))

        self.label_C = tf.reshape(self.label_holder, [1, 208, 208, 2])
        self.label_Grad = tf.cast(tf.greater(self.im_gradient(self.label_C), self.contour_th), tf.float32)
        self.label_Grad = tf.cast(tf.greater(tf.reduce_sum(self.im_gradient(self.label_C),
                                                           reduction_indices=3, keep_dims=True),
                                             self.contour_th), tf.float32)

        self.C_IoU_LOSS = self.Loss_IoU(self.Prob_Grad, self.label_Grad)

        # self.Contour_Loss = self.Loss_Contour(self.Prob_Grad, self.label_Grad)

        #Loss Function
        self.Loss_Mean = self.C_IoU_LOSS \
                         + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Score,
                                                                                  labels=self.label_holder))

        self.correct_prediction = tf.equal(tf.argmax(self.Score,1), tf.argmax(self.label_holder, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def Conv_2d(self, input_, shape, stddev, name, padding='SAME'):
        with tf.compat.v1.variable_scope(name) as scope:
            W = tf.compat.v1.get_variable('W',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

            conv = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding=padding)

            b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
            conv = tf.nn.bias_add(conv, b)

            return conv

    def Deconv_2d(self, input_, output_shape,
                  k_s=3, st_s=2, stddev=0.01, padding='SAME', name="deconv2d"):
        with tf.compat.v1.variable_scope(name):
            W = tf.compat.v1.get_variable('W',
                                shape=[k_s, k_s, output_shape[3], input_.get_shape()[3]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

            deconv = tf.nn.conv2d_transpose(input_, W, output_shape=output_shape,
                                            strides=[1, st_s, st_s, 1], padding=padding)

            b = tf.compat.v1.get_variable('b', [output_shape[3]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, b)

        return deconv

    def Contrast_Layer(self, input_, k_s=3):
        h_s = k_s / 2
        return tf.subtract(input_, tf.nn.avg_pool2d(tf.pad(input_, [[0, 0], [np.int(h_s), np.int(h_s)], [np.int(h_s), np.int(h_s)], [0, 0]], 'SYMMETRIC'),
                                                  ksize=[1, k_s, k_s, 1], strides=[1, 1, 1, 1], padding='VALID'))

    def sobel_filter(self):
        fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
        fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)

        fx = np.stack((fx, fx), axis=2)
        fy = np.stack((fy, fy), axis=2)

        fx = np.reshape(fx, (3, 3, 2, 1))
        fy = np.reshape(fy, (3, 3, 2, 1))

        tf_fx = tf.Variable(tf.constant(fx))
        tf_fy = tf.Variable(tf.constant(fy))

        return tf_fx, tf_fy

    def im_gradient(self, im):
        gx = tf.nn.depthwise_conv2d(tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC'),
                                    self.sobel_fx, [1, 1, 1, 1], padding='VALID')
        gy = tf.nn.depthwise_conv2d(tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC'),
                                    self.sobel_fy, [1, 1, 1, 1], padding='VALID')
        return tf.sqrt(tf.add(tf.square(gx), tf.square(gy)))

    def Loss_IoU(self, pred, gt):
        inter = tf.reduce_sum(tf.multiply(pred, gt))
        union = tf.add(tf.reduce_sum(tf.square(pred)), tf.reduce_sum(tf.square(gt)))

        if inter == 0:
            return 0
        else:
            return 1 - (2*(inter+1)/(union + 1))

    def Loss_Contour(self, pred, gt):
        return tf.reduce_mean(-gt*tf.log(pred+0.00001) - (1-gt)*tf.log(1-pred+0.00001))

    def L2(self, tensor, wd=0.0005):
        return tf.mul(tf.nn.l2_loss(tensor), wd, name='L2-Loss')

    def GCN1(self, gcn):    
       print(gcn)
       gcn1 = self.Conv_2d(gcn, [7, 1, 64, fea_dim], 0.01, padding='SAME', name='GCN1_11')
       print(gcn1.shape)       
       gcn1 = self.Conv_2d(gcn1, [1, 7, fea_dim, 64], 0.01, padding='SAME', name='GCN1_12')
       print(gcn1.shape)
       gcn2 = self.Conv_2d(gcn, [1, 7, 64, fea_dim], 0.01, padding='SAME', name='GCN1_21')
       print(gcn2.shape)       
       gcn2 = self.Conv_2d(gcn2, [7, 1, fea_dim, 64], 0.01, padding='SAME', name='GCN1_22')
       print(gcn2.shape)      
       gcn3 = gcn1 + gcn2
       print(gcn3.shape)
       return gcn3 

    def GCN2(self, gcn):    
       print(gcn)
       gcn1 = self.Conv_2d(gcn, [7, 1, 128, fea_dim], 0.01, padding='SAME', name='GCN2_11')
       print(gcn1.shape)       
       gcn1 = self.Conv_2d(gcn1, [1, 7, fea_dim, 128], 0.01, padding='SAME', name='GCN2_12')
       print(gcn1.shape)
       gcn2 = self.Conv_2d(gcn, [1, 7, 128, fea_dim], 0.01, padding='SAME', name='GCN2_21')
       print(gcn2.shape)       
       gcn2 = self.Conv_2d(gcn2, [7, 1, fea_dim, 128], 0.01, padding='SAME', name='GCN2_22')
       print(gcn2.shape)      
       gcn3 = gcn1 + gcn2
       print(gcn3.shape)
       return gcn3 
       
    def GCN3(self, gcn):    
       print(gcn)
       gcn1 = self.Conv_2d(gcn, [7, 1, 256, fea_dim], 0.01, padding='SAME', name='GCN3_11')
       print(gcn1.shape)       
       gcn1 = self.Conv_2d(gcn1, [1, 7, fea_dim, 256], 0.01, padding='SAME', name='GCN3_12')
       print(gcn1.shape)
       gcn2 = self.Conv_2d(gcn, [1, 7, 256, fea_dim], 0.01, padding='SAME', name='GCN3_21')
       print(gcn2.shape)       
       gcn2 = self.Conv_2d(gcn2, [7, 1, fea_dim, 256], 0.01, padding='SAME', name='GCN3_22')
       print(gcn2.shape)      
       gcn3 = gcn1 + gcn2
       print(gcn3.shape)
       return gcn3 
       
    def GCN4(self, gcn):    
       print(gcn)
       gcn1 = self.Conv_2d(gcn, [7, 1, 512, fea_dim], 0.01, padding='SAME', name='GCN4_11')
       print(gcn1.shape)       
       gcn1 = self.Conv_2d(gcn1, [1, 7, fea_dim, 512], 0.01, padding='SAME', name='GCN4_12')
       print(gcn1.shape)
       gcn2 = self.Conv_2d(gcn, [1, 7, 512, fea_dim], 0.01, padding='SAME', name='GCN4_21')
       print(gcn2.shape)       
       gcn2 = self.Conv_2d(gcn2, [7, 1, fea_dim, 512], 0.01, padding='SAME', name='GCN4_22')
       print(gcn2.shape)      
       gcn3 = gcn1 + gcn2
       print(gcn3.shape)
       return gcn3 
       
    def GCN5(self, gcn):    
       print(gcn)
       gcn1 = self.Conv_2d(gcn, [7, 1, 512, fea_dim], 0.01, padding='SAME', name='GCN5_11')
       print(gcn1.shape)       
       gcn1 = self.Conv_2d(gcn1, [1, 7, fea_dim, 512], 0.01, padding='SAME', name='GCN5_12')
       print(gcn1.shape)
       gcn2 = self.Conv_2d(gcn, [1, 7, 512, fea_dim], 0.01, padding='SAME', name='GCN5_21')
       print(gcn2.shape)       
       gcn2 = self.Conv_2d(gcn2, [7, 1, fea_dim, 512], 0.01, padding='SAME', name='GCN5_22')
       print(gcn2.shape)      
       gcn3 = gcn1 + gcn2
       print(gcn3.shape)
       return gcn3 

    def BR501(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR501_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR501_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3
    
    def BR502(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR502_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR502_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3
    
    def BR503(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [1, 1, fea_dim, fea_dim], 0.01, padding='SAME', name='BR503_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [1, 1, fea_dim, fea_dim], 0.01, padding='SAME', name='BR503_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3
      
    def BR1(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, 64, 64], 0.01, padding='SAME', name='BR1_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, 64, 64], 0.01, padding='SAME', name='BR1_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3
   
    def BR2(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, 128, 128], 0.01, padding='SAME', name='BR2_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, 128, 128], 0.01, padding='SAME', name='BR2_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3
    
    def BR3(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, 256, 256], 0.01, padding='SAME', name='BR3_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, 256, 256], 0.01, padding='SAME', name='BR3_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3
   
    def BR4(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, 512, 512], 0.01, padding='SAME', name='BR4_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, 512, 512], 0.01, padding='SAME', name='BR4_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3
       
    def BR5(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, 512, 512], 0.01, padding='SAME', name='BR5_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, 512, 512], 0.01, padding='SAME', name='BR5_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3 
       
       
    def BR11(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR11_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR11_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3
   
    def BR21(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR21_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR21_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3
    
    def BR31(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR31_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR31_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3
   
    def BR41(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR41_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR41_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3
       
    def BR51(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR51_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR51_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3 
       
    def BR52(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR52_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, fea_dim, fea_dim], 0.01, padding='SAME', name='BR52_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3 
       
    def BR42(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, fea_dim*2, fea_dim*2], 0.01, padding='SAME', name='BR42_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, fea_dim*2, fea_dim*2], 0.01, padding='SAME', name='BR42_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3 
       
    def BR32(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, fea_dim*3, fea_dim*3], 0.01, padding='SAME', name='BR32_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, fea_dim*3, fea_dim*3], 0.01, padding='SAME', name='BR32_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3 
       
    def BR22(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, fea_dim*4, fea_dim*4], 0.01, padding='SAME', name='BR22_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, fea_dim*4, fea_dim*4], 0.01, padding='SAME', name='BR22_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3 
    
    def BRLF(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, fea_dim*5, fea_dim*5], 0.01, padding='SAME', name='BRLF_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, fea_dim*5, fea_dim*5], 0.01, padding='SAME', name='BRLF_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3 

    def BRLS(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [3, 3, 2, 2], 0.01, padding='SAME', name='BRLS_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [3, 3, 2, 2], 0.01, padding='SAME', name='BRLS_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3 

    def BRGS(self, br):    
       print(br)
       br1 = tf.nn.relu(self.Conv_2d(br, [1, 1, 2, 2], 0.01, padding='SAME', name='BRGS_1'))
       print(br1.shape)
       br2 = self.Conv_2d(br1, [1, 1, 2, 2], 0.01, padding='SAME', name='BRGS_2')
       print(br2.shape)
       br3 = br + br2
       print(br3.shape)
       return br3 
             



if __name__ == "__main__":

    img = cv2.imread("dataset/MSRA-B/image/0_1_1339.jpg")

    h, w = img.shape[0:2]
    img = cv2.resize(img, (img_size,img_size)) - vgg16.VGG_MEAN
    img = img.reshape((1, img_size, img_size, 3))

    label = cv2.imread("dataset/MSRA-B/annotation/0_1_1339.png")[:, :, 0]
    label = cv2.resize(label, (label_size, label_size))
    label = label.astype(np.float32) / 255
    label = np.stack((label, 1-label), axis=2)
    label = np.reshape(label, [-1, 2])

    sess = tf.Session()

    model = Model()
    model.build_model()

    max_grad_norm = 1
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(model.C_IoU_LOSS, tvars), max_grad_norm)
    opt = tf.train.AdamOptimizer(1e-5)
    optimizer = opt.apply_gradients(zip(grads, tvars))

    sess.run(tf.global_variables_initializer())

    for i in xrange(400):
        _, C_IoU_LOSS = sess.run([optimizer, model.C_IoU_LOSS],
                                 feed_dict={model.input_holder: img,
                                            model.label_holder: label})

        print('[Iter %d] Contour Loss: %f' % (i, C_IoU_LOSS))

    boundary, gt_boundary = sess.run([model.Prob_Grad, model.label_Grad],
                                     feed_dict={model.input_holder: img,
                                                model.label_holder: label})

    boundary = np.squeeze(boundary)
    boundary = cv2.resize(boundary, (w, h))

    gt_boundary = np.squeeze(gt_boundary)
    gt_boundary = cv2.resize(gt_boundary, (w, h))

    #cv2.imshow('boundary', np.uint8(boundary*255))
    #cv2.imshow('boundary_gt', np.uint8(gt_boundary*255))
    
    save_name = ("imgB.png")
    cv2.imwrite(save_name, (boundary*255).astype(np.uint8))
    save_name2 = ("GtB.png")
    cv2.imwrite(save_name2, (gt_boundary*255).astype(np.uint8))

    cv2.waitKey()
