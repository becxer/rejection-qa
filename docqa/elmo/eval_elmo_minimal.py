import argparse
import json
from os.path import join

import nltk
import tensorflow as tf

from docqa.data_processing.qa_training_data import ParagraphAndQuestionDataset, ContextLenKey
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.data_processing.word_vectors import load_word_vector_file
from docqa.dataset import ClusteredBatcher
from docqa.model_dir import ModelDir
from docqa.squad.build_squad_dataset import parse_squad_data
from docqa.squad.squad_data import split_docs
from docqa.utils import ResourceLoader

import matplotlib.pyplot as plt
import numpy as np

"""
Used to submit our official SQuAD scores via codalab
"""


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data")
    parser.add_argument("output_data")

    parser.add_argument("--plot_dir", type=str, default=None)

    parser.add_argument("--model_dir", type=str, default="/tmp/model/document-qa")
    parser.add_argument("--lm_dir", type=str, default="/home/castle/data/lm/squad-context-concat-skip")
    parser.add_argument("--glove_dir", type=str, default="/home/castle/data/glove")

    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=30)
    parser.add_argument("--ema", action="store_true")
    args = parser.parse_args()

    input_data = args.input_data
    output_path = args.output_data
    model_dir = ModelDir(args.model_dir)
    nltk.data.path.append("nltk_data")

    print("Loading data")
    docs = parse_squad_data(input_data, "", NltkAndPunctTokenizer(), False)
    pairs = split_docs(docs)
    dataset = ParagraphAndQuestionDataset(pairs, ClusteredBatcher(args.batch_size, ContextLenKey(), False, True))

    print("Done, init model")
    model = model_dir.get_model()
    # small hack, just load the vector file at its expected location rather then using the config location
    loader = ResourceLoader(lambda a, b: load_word_vector_file(join(args.glove_dir, "glove.840B.300d.txt"), b))
    lm_model = model.lm_model
    basedir = args.lm_dir
    plotdir = args.plot_dir

    lm_model.lm_vocab_file = join(basedir, "squad_train_dev_all_unique_tokens.txt")
    lm_model.options_file = join(basedir, "options_squad_lm_2x4096_512_2048cnn_2xhighway_skip.json")
    lm_model.weight_file = join(basedir, "squad_context_concat_lm_2x4096_512_2048cnn_2xhighway_skip.hdf5")
    lm_model.embed_weights_file = None

    model.set_inputs([dataset], loader)

    print("Done, building graph")
    sess = tf.Session()
    with sess.as_default():
        pred = model.get_prediction()
    best_span = pred.get_best_span(17)[0]

    if plotdir != None:
        start_logits_op, end_logits_op = pred.get_logits()

    all_vars = tf.global_variables() + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
    dont_restore_names = {x.name for x in all_vars if x.name.startswith("bilm")}
    print(sorted(dont_restore_names))
    vars = [x for x in all_vars if x.name not in dont_restore_names]

    print("Done, loading weights")
    checkpoint = model_dir.get_best_weights()
    if checkpoint is None:
        print("Loading most recent checkpoint")
        checkpoint = model_dir.get_latest_checkpoint()
    else:
        print("Loading best weights")

    saver = tf.train.Saver(vars)
    saver.restore(sess, checkpoint)

    if args.ema:
        ema = tf.train.ExponentialMovingAverage(0)
        saver = tf.train.Saver({ema.average_name(x): x for x in tf.trainable_variables()})
        saver.restore(sess, checkpoint)

    sess.run(tf.variables_initializer([x for x in all_vars if x.name in dont_restore_names]))

    print("Done, starting evaluation")
    out = {}
    for i, batch in enumerate(dataset.get_epoch()):
        if args.n is not None and i == args.n:
            break
        print("On batch size [%d], now in %d th batch" % (args.batch_size, i +1))
        enc = model.encode(batch, False)
        if plotdir != None:
            spans, start_logits, end_logits = sess.run([best_span, start_logits_op, end_logits_op], feed_dict=enc)
            for bi, point in enumerate(batch):
                q = ' '.join(point.question)
                c = point.paragraph.get_context()
                gt = ' | '.join(point.answer.answer_text)
                s, e = spans[bi]
                pred = point.get_original_text(s, e)
                start_dist = start_logits[bi]
                end_dist = end_logits[bi]
                c_interval = np.arange(0.0, start_dist.shape[0], 1)
                c_label = c
                plt.figure(1)
                plt.subplot(211)
                plt.plot(c_interval, start_dist, color='r')
                plt.title("Q : " + q + " // A : " + gt, fontsize=9)
                plt.text(0, 0, r'Predict : %s [%d:%d]' % (pred, s, e), color='b')
                axes = plt.gca()
                axes.set_ylim([-20, 20])

                plt.subplot(212)
                plt.plot(c_interval, end_dist, color='g')
                plt.xticks(c_interval, c_label, rotation=90, fontsize=5)
                axes = plt.gca()
                axes.set_ylim([-20, 20])
                plt.show()

            break
        else:
            spans = sess.run(best_span, feed_dict=enc)

        for (s, e), point in zip(spans, batch):
            out[point.question_id] = point.get_original_text(s, e)

    sess.close()

    print("Done, saving")
    with open(output_path, "w") as f:
        json.dump(out, f)

    print("Mission accomplished!")


if __name__ == "__main__":
    run()



