# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8') #gb2312
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)

import tensorflow as tf
import numpy as np
from a1_dual_bilstm_cnn_model import DualBilstmCnnModel
from data_util_test import load_vocabulary,load_test_data,get_label_by_logits,write_predict_result
import os
#import word2vec

#configuration
FLAGS=tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string("tokenize_style",'char',"tokenize sentence in char,word,or pinyin.default is char") #to tackle miss typed words
#tf.app.flags.DEFINE_string("ckpt_dir","dual_bilstm_char_checkpoint/","checkpoint location for the model")
#tf.app.flags.DEFINE_string("model","dual_bilstm","which model to use:dual_bilstm_cnn,dual_bilstm,dual_cnn.default is:dual_bilstm_cnn")
#tf.app.flags.DEFINE_string("name_scope","cnn","name scope value.")

tf.app.flags.DEFINE_integer("embed_size",50,"embedding size") #128
tf.app.flags.DEFINE_integer("num_filters", 10, "number of filters") #32
tf.app.flags.DEFINE_integer("sentence_len",21,"max sentence length. length should be divide by 3, which is used by k max pooling.") #40
tf.app.flags.DEFINE_string("similiarity_strategy",'additive',"similiarity strategy: additive or multiply. default is additive") #to tackle miss typed words
tf.app.flags.DEFINE_string("max_pooling_style",'chunk_max_pooling',"max_pooling_style:max_pooling,k_max_pooling,chunk_max_pooling. default: chunk_max_pooling") #extract top k feature instead of max feature(max pooling)

tf.app.flags.DEFINE_integer("top_k", 3, "value of top k")
tf.app.flags.DEFINE_string("traning_data_path","./data/atec_nlp_sim_train.csv","path of traning data.")
tf.app.flags.DEFINE_integer("vocab_size",60000,"maximum vocab size.") #80000
tf.app.flags.DEFINE_float("learning_rate",0.0001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 4, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean("use_pretrained_embedding",False,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("word2vec_model_path","word2vec.bin","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_float("dropout_keep_prob", 1.0, "dropout keep probability")


#siamese
# 注意将'1524812987'更换成自己实际的目录名称
tf.flags.DEFINE_string("model", "runs/1526866723/checkpoints/model-20000", "Load trained model checkpoint (Default: None)")

filter_sizes=[2,3,4]

#def main(_):
def predict_siamese(inpath):
    logits_result=None
    checkpoint_file = FLAGS.model
    graph = tf.Graph()
    with graph:
        print("1.load vocabulary...")
        vocabulary_word2index, vocabulary_index2label= load_vocabulary(FLAGS.traning_data_path,FLAGS.vocab_size,
                                                        name_scope=name_scope,tokenize_style=tokenize_style)
        vocab_size = len(vocabulary_word2index);print(model_name+".vocab_size:",vocab_size);num_classes=len(vocabulary_index2label);print("num_classes:",num_classes)
        print("2.load data....")
        lineno_list, X1, X2,BLUESCORE=load_test_data(inpath, vocabulary_word2index, FLAGS.sentence_len, tokenize_style=tokenize_style)
        length_data_mining_features = len(BLUESCORE[0])
        print("length_data_mining_features:",length_data_mining_features)

        print("3.construct model...")
        #2.create session.
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
            input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/distance").outputs[0]

            accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

            sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]

            # emb = graph.get_operation_by_name("embedding/W").outputs[0]
            # embedded_chars = tf.nn.embedding_lookup(emb,input_x)
            # Generate batches for one epoch
            # 3.feed data & training
            number_of_test_data = len(X1)
            print(model_name + ".number_of_test_data:", number_of_test_data)
            batch_size = FLAGS.batch_size
            iteration = 0
            divide_equally = (number_of_test_data % batch_size == 0)
            steps = 0
            if divide_equally:
                steps = int(number_of_test_data / batch_size)
            else:
                steps = int(number_of_test_data / batch_size) + 1

            print("steps:", steps)
            start = 0
            end = 0
            logits_result = np.zeros((number_of_test_data, len(vocabulary_index2label)))
            for i in range(steps):
                print("i:", i)
                start = i * batch_size
                if i != steps or divide_equally:
                    end = (i + 1) * batch_size
                    # feed_dict = {model.input_x1: X1[start:end], model.input_x2: X2[start:end],
                    #              model.input_bluescores: BLUESCORE[start:end],
                    #              model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    #              model.iter: iteration, model.tst: not FLAGS.is_training}
                    feed_dict={input_x1: X1[start:end], input_x2: X2[start:end],dropout_keep_prob: 1.0}
                    print(i * batch_size, end)
                else:
                    end = number_of_test_data - (batch_size * int(number_of_test_data % batch_size))
                    # feed_dict = {model.input_x1: X1[start:end], model.input_x2: X2[start:end],
                    #              model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    #              model.iter: iteration, model.tst: not FLAGS.is_training}
                    feed_dict = {input_x1: X1[start:end], input_x2: X2[start:end], dropout_keep_prob: 1.0}
                    print("start:", i * batch_size, ";end:", end)
                logits_batch = sess.run(sim, feed_dict)  # [batch_size,num_classes]
                logits_result[start:end] = logits_batch

        print("logits_result:", logits_result)
        return logits_result, lineno_list, vocabulary_index2label


if __name__ == "__main__":
    #tf.app.run()
    pass