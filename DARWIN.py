from sys import stdout
from enum import Enum
from datetime import timedelta
import numpy as np
import pandas
from sklearn import preprocessing
import pandas as pd

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.streaming.importer.csv import importer as csv_stream_importer
from river.drift import ADWIN

pandas.options.mode.chained_assignment = None  # default='warn'
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Bidirectional
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, History
from tensorflow.keras.layers import Input, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope
from os import path, makedirs
from time import perf_counter
import time
import tensorflow as tf
import pickle
from gensim.models import Word2Vec
from anytree import Node
import memory_tree as mem_tree
from sklearn.metrics import f1_score

CASE_ID_KEY = 'case:concept:name'
ACTIVITY_KEY = 'concept:name'
TIME = 'time:timestamp'
FINAL_ACTIVITY = '_END_'
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

space = {'cellsize': scope.int(hp.loguniform('cellsize', np.log(10), np.log(150))),
         'dropout': hp.uniform("dropout", 0, 1),
         'batch_size': hp.choice('batch_size', [5, 6, 7, 8, 9, 10]),
         'vector_size': hp.choice('vector_size', [4, 5, 6, 7, 8, 9, 10]),
         'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01)),
         'n_layers': hp.choice('n_layers', [
             {'n_layers': 1},
             {'n_layers': 2, 'cellsize22': scope.int(hp.loguniform('cellsize22', np.log(10), np.log(150)))},
             {'n_layers': 3, 'cellsize32': scope.int(hp.loguniform('cellsize32', np.log(10), np.log(150))),
              'cellsize33': scope.int(hp.loguniform('cellsize33', np.log(10), np.log(150)))}
         ])
         }


class DriftDiscoverMode(Enum):
    STATIC = 1
    ADWIN = 2


class Darwin:

    @staticmethod
    def generate_csv(log_name, case_id=CASE_ID_KEY, activity=ACTIVITY_KEY, timestamp='time:timestamp'):
        csv_path = path.join('eventlog', 'CSV', log_name + '.csv')
        if not path.isfile(csv_path):
            print('Generating CSV file from XES log...')
            xes_path = path.join('eventlog', 'XES', log_name)
            xes_path += '.xes.gz' if path.isfile(xes_path + '.xes.gz') else '.xes'
            log = xes_importer.apply(xes_path, variant=xes_importer.Variants.LINE_BY_LINE)
            for trace in log:
                trace.append({activity: FINAL_ACTIVITY, timestamp: trace[-1][timestamp] + timedelta(seconds=1)})
            dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
            dataframe = dataframe.filter(items=[activity, timestamp, case_id]).sort_values(timestamp, kind='mergesort')
            dataframe = dataframe.rename(columns={activity: ACTIVITY_KEY, case_id: CASE_ID_KEY})
            makedirs(path.dirname(csv_path), exist_ok=True)
            dataframe.to_csv(csv_path, index=False)
        folder_results = path.join('results', log_name)
        folder_models = path.join('models', log_name, 'static')
        folder_params = path.join('params', log_name, 'static')
        folder_time = path.join('time', log_name)
        folder_log = path.join('log', log_name)
        hypopt_log = path.join('hypopt', log_name)
        folder_label = path.join('label', log_name)
        folder_w2v = path.join('w2v', log_name)
        folder_w2v_dict = path.join('w2v_dict', log_name)

        # make folders for save results
        makedirs(folder_results, exist_ok=True)
        makedirs(folder_models, exist_ok=True)
        makedirs(folder_params, exist_ok=True)
        makedirs(folder_time, exist_ok=True)
        makedirs(folder_log, exist_ok=True)
        makedirs(hypopt_log, exist_ok=True)
        makedirs(folder_label, exist_ok=True)
        makedirs(folder_w2v, exist_ok=True)
        makedirs(folder_w2v_dict, exist_ok=True)

    @staticmethod
    def get_cut_value(log_name):
        csv_path = path.join('eventlog', 'CSV', log_name)
        log = pd.read_csv(csv_path + '.csv')
        cut_val = (len(log) // 100) * 10
        print(cut_val)
        return cut_val

    @staticmethod
    def get_key(val, my_dict):
        for key, value in my_dict.items():
            if val == value:
                return key

    @staticmethod
    def replace_char(ele):
        ele = ele.replace(' ', '')
        ele = ele.replace('-', '')
        ele = ele.replace('+', '')
        ele = ele.replace('_', '')
        ele = ele.replace('.', '')
        ele = ele.replace(':', '')
        ele = ele.replace('(', '')
        ele = ele.replace(')', '')
        return ele

    def __init__(self, log_name, drift_discover_algorithm=DriftDiscoverMode.STATIC, model_update='S',
                 train_strategy='RT'):
        """
        Metodo costruttore
        :param log_name: nome del file CSV contenente lo stream di eventi
        :param drift_discover_algorithm: approccio utilizzato per la scoperta dei concept drift
        """
        self.log_name = log_name
        self.processed_events = 0
        self.adwin = None
        self.drift_discover_algorithm = drift_discover_algorithm
        self.le = preprocessing.LabelEncoder()
        self.cut = self.get_cut_value(self.log_name)
        self.model_update = model_update

        self.count_model = 0
        self.prefix_list = []
        self.next_act_list = []
        self.best_params = {}
        self.vec_dim = 0
        self.train_stragegy = train_strategy

        self.best_score = np.inf
        self.best_model = None
        self.best_time = 0
        self.best_numparameters = 0
        self.w2v_model = None
        self.temp_label_int = []
        self.list_emb = []
        self.word_vec_dict = {}

    def trace_conversion(self, w2v_dict, traces, vec_size):
        lststr = ' '.join([self.replace_char(str(elem)) for elem in traces])
        temp_traces = []
        temp_traces.append(lststr)

        temp_traces_int = []
        for l in temp_traces:
            l = l.split()
            win = l[-4:]
        temp_traces_int.append(win)

        pad = pad_sequences(temp_traces_int, maxlen=4, padding='pre', dtype=object, value='_PAD_')
        list_emb = []
        for l in pad:
            list_emb_temp = []
            for t in l:
                embed_vector = w2v_dict.get(t)
                if embed_vector is not None:
                    list_emb_temp.append(embed_vector)
                else:
                    list_emb_temp.append(np.zeros(shape=(vec_size)))
            list_emb.append(list_emb_temp)
        pt_int = np.array(list_emb)
        return pt_int

    def make_prediction(self, model, pt_int, labeling):
        y_pred = model.predict((pt_int), verbose=0)
        y_pred = y_pred.argmax(axis=1)
        y_pred = labeling.inverse_transform(y_pred)
        return y_pred[0]

    def get_model(self, params):
        n_layers = int(params["n_layers"]["n_layers"])

        self.vec_dim = 2 ** (params["vector_size"])
        self.opt_build_w2v()

        y_training = self.le.fit_transform(self.temp_label_int)
        x_training = np.asarray(self.list_emb)
        outsize = len(np.unique(y_training))
        y_training = to_categorical(y_training)

        input_act = Input(shape=(4, self.vec_dim), dtype='float32', name='input_act')

        x = (tf.keras.layers.LSTM(int(params["cellsize"]),
                                  kernel_initializer='glorot_uniform',
                                  return_sequences=(n_layers != 1)
                                  ))(input_act)

        x = tf.keras.layers.Dropout(params["dropout"])(x)

        for i in range(2, n_layers + 1):
            return_sequences = (i != n_layers)
            x = (tf.keras.layers.LSTM(int(params["n_layers"]["cellsize%s%s" % (n_layers, i)]),
                                      kernel_initializer='glorot_uniform',
                                      return_sequences=return_sequences,
                                      ))(x)

            x = tf.keras.layers.Dropout(params["dropout"])(x)

        out_a = Dense(outsize, activation='softmax', kernel_initializer='glorot_uniform', name='output_a')(x)

        model = Model(inputs=input_act, outputs=out_a)
        opt = Adam(learning_rate=params["learning_rate"])
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

        return model, x_training, y_training

    def train_and_evaluate_model(self, params):
        start_time = perf_counter()

        model, x_training, y_training = self.get_model(params)

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)

        history = model.fit(x_training, y_training,
                            validation_split=0.2, verbose=0,
                            callbacks=[early_stopping, lr_reducer],
                            batch_size=2 ** params['batch_size'], epochs=200)

        scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
        score = min(scores)
        end_time = perf_counter()

        if self.best_score > score:
            self.best_score = score
            self.best_model = model
            self.best_numparameters = model.count_params()
            self.best_time = end_time - start_time

            if self.drift_discover_algorithm == DriftDiscoverMode.ADWIN:
                model.save(
                    "models/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model) + ".h5")
            elif self.drift_discover_algorithm == DriftDiscoverMode.STATIC:
                model.save("models/" + self.log_name + "/static/" + "generate_" + self.log_name + '_' + str(
                    self.count_model) + ".h5")

            output = open(
                "w2v_dict/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model) + ".pkl", 'wb')
            pickle.dump(self.word_vec_dict, output)
            output.close()
            self.w2v_model.save(
                "w2v/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model) + ".h5")
            output = open("label/" + "generate_" + self.log_name + '_' + str(self.count_model) + ".pkl", 'wb')
            pickle.dump(self.le, output)
            output.close()

        return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(history.history['loss']),
                'n_params': model.count_params(), 'time': end_time - start_time}

    def fine_tuning_neural_network(self):
        x_training = (self.list_emb)
        y_training = self.le.fit_transform(self.temp_label_int)
        outsize = len(np.unique(y_training))
        y_training = to_categorical(y_training)

        model_old = load_model(
            "models/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model - 1) + ".h5")

        new_input = model_old.input
        x = model_old.layers[-2].output
        out_a = Dense(outsize, activation='softmax', kernel_initializer='glorot_uniform', name='output_a2')(x)
        model_new = Model(inputs=new_input, outputs=out_a)

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)

        with open("params/" + self.log_name + "/" + self.log_name + ".pickle", 'rb') as pickle_file:
            params = pickle.load(pickle_file)
        opt = Adam(learning_rate=params["learning_rate"])

        model_new.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
        model_new.fit(x_training, y_training, validation_split=0.2, verbose=0,
                      callbacks=[early_stopping, lr_reducer],
                      batch_size=int(2 ** params['batch_size']), epochs=200)
        model_new.save("models/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model) + ".h5")
        output = open("label/" + "generate_" + self.log_name + '_' + str(self.count_model) + ".pkl", 'wb')
        pickle.dump(self.le, output)
        output.close()

    def retrain_neural_network_model(self):
        with open("params/" + self.log_name + "/" + self.log_name + ".pickle", 'rb') as pickle_file:
            params = pickle.load(pickle_file)

        n_layers = int(params["n_layers"]["n_layers"])

        x_training = np.asarray(self.list_emb)
        y_training = self.le.fit_transform(self.temp_label_int)
        outsize = len(np.unique(y_training))
        y_training = to_categorical(y_training)

        input_act = Input(shape=(4, 2 ** params['vector_size']), dtype='float32', name='input_act')

        x = (tf.keras.layers.LSTM(int(params["cellsize"]),
                                  kernel_initializer='glorot_uniform',
                                  return_sequences=(n_layers != 1),
                                  ))(input_act)

        x = tf.keras.layers.Dropout(params["dropout"])(x)

        for i in range(2, n_layers + 1):
            return_sequences = (i != n_layers)
            x = (tf.keras.layers.LSTM(int(params["n_layers"]["cellsize%s%s" % (n_layers, i)]),
                                      kernel_initializer='glorot_uniform',
                                      return_sequences=return_sequences,
                                      ))(x)

            x = tf.keras.layers.Dropout(params["dropout"])(x)

        out_a = Dense(outsize, activation='softmax', kernel_initializer='glorot_uniform', name='output_a')(x)
        opt = Adam(learning_rate=params["learning_rate"])
        model = Model(inputs=input_act, outputs=out_a)

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)
        model.fit(x_training, y_training,
                  validation_split=0.2, verbose=0,
                  callbacks=[early_stopping, lr_reducer],
                  batch_size=2 ** params['batch_size'], epochs=200)

        model.save("models/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model) + ".h5")

        output = open("label/" + "generate_" + self.log_name + '_' + str(self.count_model) + ".pkl", 'wb')
        pickle.dump(self.le, output)
        output.close()

    def opt_build_w2v(self):
        temp_traces = []
        for k in self.prefix_list:
            listToStr = ' '.join([self.replace_char(str(elem)) for elem in k[1]])
            temp_traces.append(listToStr)

        self.temp_label_int = []
        for l in self.next_act_list:
            self.temp_label_int.append(l[1])

        tokenized_words = []
        for s in temp_traces:
            tokenized_words.append(s.split(' '))

        self.w2v_model = Word2Vec(vector_size=self.vec_dim, seed=seed, sg=0, min_count=1, workers=1)
        self.w2v_model.build_vocab(tokenized_words, min_count=1)
        total_examples = self.w2v_model.corpus_count
        self.w2v_model.train(tokenized_words, total_examples=total_examples, epochs=200)
        vocab = list(self.w2v_model.wv.index_to_key)

        self.word_vec_dict = {}
        for word in vocab:
            self.word_vec_dict[word] = self.w2v_model.wv.get_vector(word)

        temp_traces_int = []
        for l in temp_traces:
            l = l.split()
            win = l[-4:]
            temp_traces_int.append(win)
        pad_rev = pad_sequences(temp_traces_int, maxlen=4, padding='pre', dtype=object, value='_PAD_')

        self.list_emb = []
        for l in pad_rev:
            list_emb_temp = []
            for t in l:
                embed_vector = self.word_vec_dict.get(t)
                if embed_vector is not None:
                    list_emb_temp.append(embed_vector)
                else:
                    list_emb_temp.append(np.zeros(shape=(self.vec_dim)))
            self.list_emb.append(list_emb_temp)
        self.list_emb = np.asarray(self.list_emb, dtype=float)

    def build_w2v(self):
        temp_traces = []
        for k in self.prefix_list:
            listToStr = ' '.join([self.replace_char(str(elem)) for elem in k[1]])
            temp_traces.append(listToStr)

        self.temp_label_int = []
        for l in self.next_act_list:
            self.temp_label_int.append(l[1])

        tokenized_words = []
        for s in temp_traces:
            tokenized_words.append(s.split(' '))

        self.w2v_model = Word2Vec(vector_size=self.vec_dim, seed=seed, min_count=1, sg=0, workers=1)
        self.w2v_model.build_vocab(tokenized_words, min_count=1)
        total_examples = self.w2v_model.corpus_count
        self.w2v_model.train(tokenized_words, total_examples=total_examples, epochs=200)
        self.w2v_model.save("w2v/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model) + ".h5")
        vocab = list(self.w2v_model.wv.index_to_key)
        self.word_vec_dict = {}
        for word in vocab:
            self.word_vec_dict[word] = self.w2v_model.wv.get_vector(word)

        temp_traces_int = []
        for l in temp_traces:
            l = l.split()
            win = l[-4:]
            temp_traces_int.append(win)

        pad_rev = pad_sequences(temp_traces_int, maxlen=4, padding='pre', dtype=object, value='_PAD_')

        self.list_emb = []
        for l in pad_rev:
            list_emb_temp = []
            for t in l:
                embed_vector = self.word_vec_dict.get(t)
                if embed_vector is not None:
                    list_emb_temp.append(embed_vector)
                else:
                    list_emb_temp.append(np.zeros(shape=(self.vec_dim)))
            self.list_emb.append(list_emb_temp)
        self.list_emb = np.asarray(self.list_emb)

        output = open("w2v_dict/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model) + ".pkl",
                      'wb')
        pickle.dump(self.word_vec_dict, output)
        output.close()

    def build_w2v_fine(self):
        temp_traces = []
        for k in self.prefix_list:
            listToStr = ' '.join([self.replace_char(str(elem)) for elem in k[1]])
            temp_traces.append(listToStr)

        self.temp_label_int = []
        for l in self.next_act_list:
            self.temp_label_int.append(l[1])

        tokenized_words = []
        for s in temp_traces:
            tokenized_words.append(s.split(' '))

        old_model = Word2Vec.load(
            "w2v/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model - 1) + ".h5")
        old_model.build_vocab(tokenized_words, min_count=1, update=True)
        old_model.train(tokenized_words, total_examples=len(tokenized_words), epochs=200)
        old_model.save("w2v/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model) + ".h5")

        vocab = list(old_model.wv.index_to_key)

        self.word_vec_dict = {}
        for word in vocab:
            self.word_vec_dict[word] = old_model.wv.get_vector(word)
        temp_traces_int = []
        for l in temp_traces:
            l = l.split()
            win = l[-4:]
            temp_traces_int.append(win)

        pad_rev = pad_sequences(temp_traces_int, maxlen=4, padding='pre', dtype=object, value='_PAD_')

        self.list_emb = []
        for l in pad_rev:
            list_emb_temp = []
            for t in l:
                embed_vector = self.word_vec_dict.get(t)
                if embed_vector is not None:
                    list_emb_temp.append(embed_vector)
                else:
                    list_emb_temp.append(np.zeros(shape=(self.vec_dim)))

            self.list_emb.append(list_emb_temp)
        self.list_emb = np.asarray(self.list_emb)

        output = open("w2v_dict/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model) + ".pkl",
                      'wb')
        pickle.dump(self.word_vec_dict, output)
        output.close()

    def time_format(self, time_stamp):
        '''
        :param time_stamp: oggetto timestamp
        :return: converte l'oggetto timestamp utile in fase di calcolo dei tempi
        '''
        try:
            date_format_str = '%Y-%m-%d %H:%M:%S.%f%z'
            conversion = datetime.strptime(time_stamp, date_format_str)
        except:
            date_format_str = '%Y-%m-%d %H:%M:%S%f%z'
            conversion = datetime.strptime(time_stamp, date_format_str)
        return conversion

    def process_stream(self):

        prediction = {}
        self.adwin = ADWIN()
        update_model = False
        over_time = False
        prediction_temp = {}
        hashtable = {}
        root = Node(id='root', name='root', parent=None, case_id=[])
        i = 0

        log_file = open('log/' + self.log_name + '/' + self.log_name + '_' + self.model_update + '.log', 'w')
        log_file.write('Processing event stream...\n')
        file_prediction = open(
            'results/' + self.log_name + '/' + self.log_name + '_' + self.model_update + '_result.csv', 'w')
        file_prediction.write('Acc,' + 'Predicted,' + 'Real' + '\n')

        print('Processing event stream...')

        stream = csv_stream_importer.apply(path.join('eventlog', 'CSV', self.log_name + '.csv'))
        get_pred_curr = 0

        for event in stream:
            case = event[CASE_ID_KEY]
            activity = event[ACTIVITY_KEY]
            event_time = event[TIME]
            check_case_id = hashtable.get(case)
            self.processed_events += 1
            if update_model == True:
                start = self.time_format(last_time_event)
                end = self.time_format(event_time)
                diff = end - start
                sec = 86400 * diff.days + diff.seconds + diff.microseconds / 1000000
                get_pred_curr = 0

                if (sec) < training_time:
                    log_file.write('\nAttention! Training time > Stream Time')
                    log_file.write('\nload previous model...')
                    over_time = True
                else:
                    update_model = False
                    over_time = False

                log_file.write('\nOLD Model -->' + "models/" + self.log_name + "/generate_" + self.log_name + '_' + str(
                    self.count_model - 1) + ".h5")
                log_file.write('\nNEW Model -->' + "models/" + self.log_name + "/generate_" + self.log_name + '_' + str(
                    self.count_model) + ".h5")
            if activity != FINAL_ACTIVITY:
                if check_case_id == None:
                    child = mem_tree.check_exist_child(root.children, activity)
                    if child == None:
                        nodo = Node(name=i, id=activity, parent=root)
                        i = i + 1
                    else:
                        nodo = child
                else:
                    x = hashtable.get(case)
                    child = mem_tree.check_exist_child(x.children, activity)
                    if child != None:
                        nodo = child
                    else:
                        father = x.id
                        grandfather = x.parent.id
                        if father == activity and grandfather != activity:
                            nodo = Node(name=i, id=activity, parent=x)
                            i = i + 1
                        elif father == activity and grandfather == activity:
                            nodo = x
                        else:
                            nodo = Node(name=i, id=activity, parent=x)
                            i = i + 1
                    traces = mem_tree.get_trace(check_case_id)
                    temp_prefix = []

                    for t in traces:
                        temp_prefix.append(t)
                    self.prefix_list.append(tuple((case, temp_prefix)))
                    self.next_act_list.append((tuple((case, activity))))
                hashtable[case] = nodo
            else:
                traces = mem_tree.get_trace(check_case_id)
                temp_prefix = []

                for t in traces:
                    temp_prefix.append(t)
                self.prefix_list.append(tuple((case, temp_prefix)))
                self.next_act_list.append((tuple((case, activity))))
                hashtable.pop(str(case))
                mem_tree.pruned_tree(check_case_id, hashtable)

            if self.processed_events == self.cut:
                    if self.drift_discover_algorithm == DriftDiscoverMode.ADWIN:

                        log_file.write('\n10% of events')
                        outfile = open('hypopt/' + self.log_name + '/' + self.log_name + '_' + str(
                            self.drift_discover_algorithm) + '.log', 'w')
                        log_file.write('Starting model selection...')
                        trials = Trials()
                        best = fmin(self.train_and_evaluate_model, space, algo=tpe.suggest, max_evals=20, trials=trials,
                                    rstate=np.random.RandomState(seed))
                        self.best_params = hyperopt.space_eval(space, best)
                        log_file.write("Best parameters:" + str(self.best_params))

                        if self.drift_discover_algorithm == DriftDiscoverMode.ADWIN:
                            with open('params/' + self.log_name + '/' + self.log_name + '.pickle', 'wb') as handle:
                                pickle.dump(self.best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        else:
                            with open('params/' + self.log_name + '/static/' + self.log_name + '.pickle',
                                      'wb') as handle:
                                pickle.dump(self.best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        outfile.write("\nHyperopt trials")
                        outfile.write("\ntid,loss,learning_rate,n_modules,batch_size,time,n_epochs,n_params,perf_time")
                        for trial in trials.trials:
                            outfile.write("\n%d,%f,%f,%d,%d,%d,%s,%d,%d,%f" % (trial['tid'],
                                                                               trial['result']['loss'],
                                                                               trial['misc']['vals']['learning_rate'][
                                                                                   0],
                                                                               int(trial['misc']['vals']['n_layers'][
                                                                                       0] + 1),
                                                                               trial['misc']['vals']['batch_size'][
                                                                                   0] + 5,
                                                                               trial['misc']['vals']['vector_size'][
                                                                                   0] + 6,
                                                                               (trial['refresh_time'] - trial[
                                                                                   'book_time']).total_seconds(),
                                                                               trial['result']['n_epochs'],
                                                                               trial['result']['n_params'],
                                                                               trial['result']['time']))

                        outfile.write("\n\nBest parameters:")
                        outfile.write("\nModel parameters: %d" % self.best_numparameters)
                        outfile.write('\nBest Time taken: %f' % self.best_time)

                    model = load_model(
                        "models/" + self.log_name + "/generate_" + self.log_name + '_' + str(self.count_model) + ".h5")

                    with open("label/" + "generate_" + self.log_name + '_' + str(self.count_model) + ".pkl",
                              'rb') as pickle_file:
                        label = pickle.load(pickle_file)

                    with open("w2v_dict/" + self.log_name + "/generate_" + self.log_name + '_' + str(
                            self.count_model) + ".pkl", 'rb') as pickle_file:
                        w2v_dict = pickle.load(pickle_file)

                    with open("params/" + self.log_name + "/" + self.log_name + ".pickle", 'rb') as pickle_file:
                        params = pickle.load(pickle_file)

                    self.vec_dim = 2 ** int(params["vector_size"])

            elif self.processed_events > self.cut:
                stdout.write(f'\r\tCurrent event: {self.processed_events}')
                if case in prediction:
                    if case in prediction_temp:
                        log_file.write('\nCase over the time-->' + str(case))
                        if prediction_temp[case][0] == activity:
                            acc = 1
                        else:
                            acc = 0
                        file_prediction.write(str(acc) + ',' + prediction_temp[case][0] + ',' + activity + '\n')
                        prediction_temp.pop(case)

                        if prediction[case][0] == activity:
                            res = 1
                        else:
                            res = 0
                    else:
                        if prediction[case][0] == activity:
                            res = 1
                        else:
                            res = 0
                        file_prediction.write(str(res) + ',' + prediction[case][0] + ',' + activity + '\n')

                    prediction.pop(str(case))

                    if self.drift_discover_algorithm == DriftDiscoverMode.ADWIN:
                        in_drift, in_warning = self.adwin.update(res)
                        if in_drift:
                            number_of_traces_to_be_removed = len(self.next_act_list) - int(self.adwin.width)
                            log_file.write('\nint(self.adwin.width)' + str(int(self.adwin.width)))
                            log_file.write('\nself.next_act_list' + str(len(self.next_act_list)))
                            log_file.write('\nself.prefix_list' + str(len(self.prefix_list)))
                            log_file.write(
                                '\nnumber_of_traces_to_be_removed' + str(number_of_traces_to_be_removed) + '\n')
                            self.prefix_list = self.prefix_list[number_of_traces_to_be_removed:]
                            self.next_act_list = self.next_act_list[number_of_traces_to_be_removed:]
                            log_file.write('\nlen self.prefix_list' + str(len(self.prefix_list)))

                            output = open("label/" + "adwin_" + self.log_name + '_' + str(self.count_model) + ".pkl",
                                          'wb')
                            pickle.dump(self.prefix_list, output)
                            output.close()

                            if update_model == False:
                                log_file.write('\nDrift detected at index -->' + str(self.processed_events))
                                start_time = time.process_time()
                                self.count_model += 1
                                if self.train_stragegy == 'FT':
                                    self.build_w2v_fine()
                                    self.fine_tuning_neural_network()
                                else:
                                    self.build_w2v()
                                    self.retrain_neural_network_model()
                                end_time = time.process_time()
                                training_time = end_time - start_time
                                update_model = True
                                last_time_event = event_time
                                old_model = load_model(
                                    "models/" + self.log_name + "/generate_" + self.log_name + '_' + str(
                                        self.count_model - 1) + '.h5')
                                model = load_model("models/" + self.log_name + "/generate_" + self.log_name + '_' + str(
                                    self.count_model) + '.h5')
                                get_pred_curr = 1
                                with open("label/" + "generate_" + self.log_name + '_' + str(
                                        self.count_model - 1) + ".pkl", 'rb') as pickle_file:
                                    old_label = pickle.load(pickle_file)

                                with open("label/" + "generate_" + self.log_name + '_' + str(self.count_model) + ".pkl",
                                          'rb') as pickle_file:
                                    label = pickle.load(pickle_file)

                                with open("w2v_dict/" + self.log_name + "/generate_" + self.log_name + '_' + str(
                                        self.count_model - 1) + ".pkl", 'rb') as pickle_file:
                                    old_w2v_dict = pickle.load(pickle_file)

                                with open("w2v_dict/" + self.log_name + "/generate_" + self.log_name + '_' + str(
                                        self.count_model) + ".pkl", 'rb') as pickle_file:
                                    w2v_dict = pickle.load(pickle_file)

                traces = (mem_tree.get_trace(nodo))
                if FINAL_ACTIVITY not in traces:
                    if update_model == False:
                        pt_int = self.trace_conversion(w2v_dict, traces, self.vec_dim)
                        y_pred2 = self.make_prediction(model, pt_int, label)
                        prediction[case] = [y_pred2]
                    else:
                        if over_time == True:
                            log_file.write('\nPrediction of Case over the time-->' + str(case))
                            pt_int = self.trace_conversion(w2v_dict, traces, self.vec_dim)
                            y_pred2 = self.make_prediction(model, pt_int, label)
                            prediction[case] = [y_pred2]

                            pt_int_old = self.trace_conversion(old_w2v_dict, traces, self.vec_dim)
                            y_pred = self.make_prediction(old_model, pt_int_old, old_label)
                            prediction_temp[case] = [y_pred]
                        elif get_pred_curr == 1:
                            pt_int_old = self.trace_conversion(old_w2v_dict, traces, self.vec_dim)
                            y_pred = self.make_prediction(old_model, pt_int_old, old_label)
                            prediction[case] = [y_pred]
        if self.drift_discover_algorithm == DriftDiscoverMode.ADWIN:
            res = pd.read_csv('results/' + self.log_name + '/' + self.log_name + '_DriftDiscoverMode.ADWIN_result.csv')
            st = '_D_'
        else:
            res = pd.read_csv('results/' + self.log_name + '/' + self.log_name + '_DriftDiscoverMode.STATIC_result.csv')
            st = '_S_'
        fres = (round(f1_score(res['Real'].to_list(), res['Predicted'].to_list(), average='macro'), 3))
        output = open("results/res" + st +self.log_name +  ".pkl",'wb')
        pickle.dump(fres, output)
        output.close()
