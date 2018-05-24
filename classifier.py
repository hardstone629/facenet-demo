import face
import tensorflow as tf
import numpy as np
import sys
import facenet
import pickle
import os
import math
from sklearn.svm import SVC

class Classifier:
    def __init__(self):
        self.model = ""
        self.imagePath = os.getcwd()+"/dataset/orignal"
    def train(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                np.random.seed(seed=666)
                dataset_tmp = facenet.get_dataset(self.imagePath)
                train_set, test_set = self.split_dataset(dataset_tmp)
                dataset = test_set
                for cls in dataset:
                    assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')
                paths,labels =facenet.get_image_paths_and_labels(dataset)
                print('Number of classes: %d' % len(dataset))
                print('Number of images: %d' % len(paths))
                
                # Load the model
                print('Loading feature extraction model')
                facenet.load_model(face.facenet_model_checkpoint)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                print('Calculating features for images')
                nrof_images = len(paths)
                batch_size = 100
                nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i*batch_size
                    end_index = min((i+1)*batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, face.image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                
                classifier_filename_exp = os.path.expanduser(face.classifier_model)
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

    def test(self,dataset):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                np.random.seed(seed=666)
                paths,labels =facenet.get_image_paths_and_labels(dataset)


    # min_nrof_images_per_class:
    # nrof_train_images_per_class:
    def split_dataset(self,dataset, min_nrof_images_per_class=20, nrof_train_images_per_class=10):
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            # Remove classes with less than min_nrof_images_per_class
            if len(paths)>=min_nrof_images_per_class:
                np.random.shuffle(paths)
                train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
                test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
        return train_set, test_set            

if __name__ == '__main__':
    classfiter = Classifier()
    classfiter.train()