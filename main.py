import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.stacklayout import StackLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image


import pandas as pd
import matplotlib.pyplot as plt
import pydot
import graphviz
import time
import io


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from lime import lime_tabular




##################################################################################################
##################################################################################################
###########  initialization



class neuralize_it(App):

    def build(self):

        self.icon = 'logo.jpg'
        panel = StackLayout(orientation='lr-tb')

        ### set buttons

        n_small = 10
        small_empty = [0]*n_small
        for i in range(n_small):
            small_empty[i] = Label( size_hint = (.03, .15) )


        n_empty = 30
        empty = [0]*n_empty
        for i in range(n_empty):
            empty[i] = Label( size_hint = (.1, .15) )
        
        intro = Label(
            text="Welcome to neuralize_it!!! \n \nThe interface for running dense \nneural networks with TensorFlow.",
            size_hint = (.2, .15) )
        logo = Image( source='logo.png',
                      pos_hint={'center_x': .5, 'center_y': .5}, size_hint = (.05, .135) )
        info_b = Button(text="About", font_size = 20, size_hint = (.1, .15))
        info_b.bind(on_press = self.pop_up_about)


        data_b = Button(text="Load Data", font_size = 20, size_hint = (.1, .05), background_color = 'red')
        data_b.bind(on_press=self.load_data)
        set_b = Button(text="Set Inputs", font_size = 20, size_hint = (.1, .05), background_color = 'lightblue')
        set_b.bind(on_press=self.set_parameters)
        train_b = Button(text="Train Model", font_size = 20, size_hint = (.1, .05), background_color = 'lightgreen')
        train_b.bind(on_press=self.train_ks)
        export_b = Button(text="Save Weights", font_size = 20, size_hint = (.1, .05), background_color = 'dodgerblue')
        export_b.bind(on_press=self.save_model)


        data_guides = Label(
            text="Place your dataset in 'data' folder, \nthe dataset should be named 'data.csv' \nand press the red button to load your data  \n \n \n",
            size_hint = (.23, .15) )
        col_l = Label(
            text="Set the 'n' number of output columns. \nThe last n columns of your dataset are \nconsidered output and the rest input. \n \n",
            size_hint = (.23, .15) )
        self.col_i = TextInput(multiline=True, size_hint = (.06, .08), text = "1", halign="auto")
        per_l = Label(
            text="The dataset is split to training and \ntesting. \nSet the percentage of training data. \nSet 1 for full dataset training. \n",
            size_hint = (.23, .15) )
        self.per_i = TextInput(multiline=True, size_hint = (.06, .08), text = "0.7", halign="auto")

        struct_l = Label(
            text="Set neural's Structure. \nSeparate layers with \ncomma or newline. \nLayer input: \n(kernels, activation function)",
            size_hint = (.2, .15) )
        self.struct_i = TextInput(multiline=True, size_hint = (.2, .15), text = "(80, relu)\n(30, linear)\n(1, sigmoid)")

        train_l = Label(
            text="Set Training inputs: \noptimizer, loss function, \nmetrics, epochs, batch_s",
            size_hint = (.2, .15) )
        self.train_i = TextInput(multiline=True, size_hint = (.2, .15), text = "Adam, mse, mae, 20, 100")

        pred_l = Label(
            text="Set x inputs for prediction: \n \n \n \n",
            size_hint = (.2, .15) )
        self.pred_i = TextInput(multiline=True, size_hint = (.2, .15), text = "0, 0, 0, 0, 0, 0\n1, 1, 1, 1, 1, 1")


        status_b = Button(text="Status", font_size = 20, size_hint = (.1, .15), background_color = 'steelblue')
        status_b.bind(on_press=self.input_status)
        self.data = 'Empty'
        self.layers_n = 'Empty'
        self.max_row = 'Empty'
        self.layers = 'Empty'
        self.train = 'Empty'

        dataprint_b = Button(text="Data", font_size = 20, size_hint = (.1, .15), background_color = 'lightcoral')
        dataprint_b.bind(on_press=self.dataprint)

        model_b = Button(text="Training\nResults", font_size = 20, size_hint = (.1, .15), background_color = 'greenyellow')
        model_b.bind(on_press=self.training_plots)
        self.model_history = 'No model trained !!!'
        self.last_training = 'No model trained !!!'

        modeluml_b = Button(text="Neural\nUML", font_size = 20, size_hint = (.1, .15), background_color = 'darkseagreen')
        modeluml_b.bind(on_press=self.neural_uml)

        pred_b = Button(text="Prediction", font_size = 20, size_hint = (.1, .15), background_color = 'magenta')
        pred_b.bind(on_press=self.predict)

        options_b = Button(text="Options", font_size = 20, size_hint = (.1, .15))
        options_b.bind(on_press=self.tf_options)


        panel.add_widget(info_b)
        panel.add_widget(intro)
        panel.add_widget(logo)
        for i in range(5):
            panel.add_widget(empty[i])

        for i in range(7):
            panel.add_widget(empty[6+i])
        panel.add_widget(data_b)
        panel.add_widget(set_b)
        panel.add_widget(train_b)
        panel.add_widget(export_b)

        panel.add_widget(data_guides)
        panel.add_widget(small_empty[0])
        panel.add_widget(col_l)
        panel.add_widget(self.col_i)
        panel.add_widget(small_empty[1])
        panel.add_widget(per_l)
        panel.add_widget(self.per_i)

        panel.add_widget(struct_l)
        panel.add_widget(self.struct_i)
        for j in range(5):
            panel.add_widget(empty[j+14])

        panel.add_widget(train_l)
        panel.add_widget(self.train_i)
        for j in range(2):
            panel.add_widget(empty[j+19])
        panel.add_widget(status_b)
        panel.add_widget(dataprint_b)
        panel.add_widget(model_b)

        panel.add_widget(pred_l)
        panel.add_widget(self.pred_i)
        for j in range(2):
            panel.add_widget(empty[j+24])
        panel.add_widget(modeluml_b)
        panel.add_widget(pred_b)
        panel.add_widget(options_b)


        return panel


    # set functions

    def pop_up_about(self, instance):
        with open('about.txt') as f:
            txt = f.read()
        pop_Wind = Popup(title="About", content=Label(text=txt), size_hint=(None,None), size=(400,400))
        pop_Wind.open()

    def load_data(self, instance):
        try:
            diff = time.time()
            self.data = pd.read_csv('data\data.csv')
            diff = - diff + time.time()
            self.loaded_txt = "Data Loaded at \n" + time.strftime('%H:%M:%S')
            loaded_label = Label( text =  "Data Loaded in " + str(diff) + "'.")
            pop_data = Popup(title="Data Loaded!!!", content=loaded_label, size_hint=(None, None), size=(400, 400))
            pop_data.open()
        except:
            loaded_label = Label( text =  "Problem with Data!!! \n\nPlease check that the dataset is \nin 'data' folder and named \n'data.csv'!!!")
            pop_data = Popup(title="Data not Loaded...", content=loaded_label, size_hint=(None, None), size=(400, 400))
            pop_data.open()

    def set_parameters(self, instance):
        try:
            temp = self.struct_i.text
            temp = temp.split(')\n')
            self.layers_n = len(temp)
            for i in range(self.layers_n):
                temp[i] = temp[i].replace('(','')
                temp[i] = temp[i].replace(')','')
            self.layers = temp
            temp = self.train_i.text.split(', ')
            temp[3] = int(temp[3])
            temp[4] = int(temp[4])
            self.train = temp
            loaded_label = Label( text =  "Layers:\n" + self.struct_i.text + ',\n\nand Inputs:\n' + self.train_i.text + ",\n\nare set!!!")
            pop_data = Popup(title="Inputs Loaded!!!", content=loaded_label, size_hint=(None, None), size=(400, 400))
            pop_data.open()
        except:
            loaded_label = Label(
                text="Problem with Inputs!!! \n\nPlease check that the inputs \nare set properly, similarly \nwith TensorFlow!!!")
            pop_data = Popup(title="Inputs not Loaded...", content=loaded_label, size_hint=(None, None), size=(400, 400))
            pop_data.open()

    def train_ks(self, instance):
        self.last_training = time.strftime('%H:%M:%S')
        try:
            model = Sequential()
            for i in range(self.layers_n):
                inputs = self.layers[i].split(', ')
                inputs[0] = int(inputs[0])
                model.add(Dense(inputs[0], activation=inputs[1]))
            if  inputs[0] != int(self.col_i.text):
                model.add(Dense(int(self.col_i.text), activation="linear"))
            model.compile(optimizer= self.train[0],
                          loss= self.train[1],
                          metrics= self.train[2])
            self.max_row = int(self.data.shape[0]*float(self.per_i.text))
            last_col = self.data.shape[1] - int(self.col_i.text)
            x = self.data.iloc[ 0:self.data.shape[0], 0:last_col].copy()
            y = self.data.iloc[ 0:self.data.shape[0], last_col:self.data.shape[1]].copy()
            self.model_history = model.fit( x, y, batch_size=self.train[3], epochs=self.train[4], validation_split=1-float(self.per_i.text), verbose=0)
            self.model = model
            loaded_label = Label( text =  "Model Trained \n" + str(self.model_history) )
            pop_data = Popup(title="Training Completed!!!", content=loaded_label, size_hint=(None, None), size=(600, 600))
            pop_data.open()
        except:
            loaded_label = Label( text =  "Model could not be trained... \n\nCheck your data or inputs!!!!")
            pop_data = Popup(title="Training could not happen...", content=loaded_label, size_hint=(None, None), size=(400, 400))
            pop_data.open()

    def input_status(self, instance):
            temp = 'Data: \n'
            if type(self.data) != str:
                buf = io.StringIO()
                self.data.info(verbose=False, buf=buf)
                temp = temp + buf.getvalue() + '\n\n'
                a = self.data.shape[0]
            else:
                temp = temp + self.data + '\n\n'
                a = 0
                b = 0

            b = 0
            if self.per_i.text != '':
                b = float(self.per_i.text)

            temp = temp + 'Inputs: \n'
            temp = temp + 'Neural Layers: ' + str(self.layers_n) + '\n'
            temp = temp + 'Training Sample size: ' + str(int(a*b)) + ' rows' + '\n\n'

            temp = temp + 'Training: \n'
            if int(self.col_i.text) > 0 and type(self.data) != str:
                temp = temp + 'x cols:  1 to ' + str(self.data.shape[1] - int(self.col_i.text)) + '\n'
                temp = temp + 'y cols: ' + str(self.data.shape[1] - int(self.col_i.text) + 1) + ' to last' + '\n'
            else:
                temp = temp + 'Not proper x-y splitting. \n'
            temp = temp + 'Training inputs: ' + str(self.train)
            loaded_label = Label( text =  temp )
            pop_data = Popup(title="Inputs Status:", content=loaded_label, size_hint=(None, None), size=(600, 600))
            pop_data.open()

    def dataprint(self, instance):
        if type(self.data) != str:
            buf = io.StringIO()
            self.data.to_string( buf = buf )
            temp = buf.getvalue()
        else:
            temp = 'No imported data!!!'
        loaded_label = Label(text=temp)
        pop_data = Popup(title="Dataset:", content=loaded_label, size_hint=(None, None), size=(600, 600))
        pop_data.open()

    def training_plots(self, instance):
        if type(self.model_history) == str:
            loaded_label = Label(text=self.model_history)
            pop_data = Popup(title="No Training Found!!!", content=loaded_label, size_hint=(None, None), size=(400, 400))
            pop_data.open()
        else:
            plt.plot(self.model_history.history[self.train[2]])
            plt.plot(self.model_history.history['val_' + self.train[2]])
            plt.title('Model trained at ' + self.last_training)
            plt.ylabel(self.train[2])
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

    def neural_uml(self, instance):
        try:
            plot_model(self.model, to_file='neural.png', show_layer_activations=True, show_shapes=False,
                                      show_layer_names=False, rankdir="LR")
            uml = Image( source='neural.png' )
            pop_data = Popup(title="UML of your trained neural:", content=uml, size=(600, 600), size_hint=(None, None) )
            pop_data.open()
        except:
            loaded_label = Label(text='No model to be printed!!! \n\n Check your training!!!')
            pop_data = Popup(title="No model to be printed!!!", content=loaded_label, size_hint=(None, None), size=(400, 400))
            pop_data.open()


    def predict(self, instance):
        try:
            temp = self.pred_i.text.split('\n')
            xf = []
            for i in temp:
                temp2 = i.split(', ')
                x = []
                for j in temp2:
                    x.append(float(j))
                xf.append(x)
            xf = pd.DataFrame(xf)
            prediction = self.model.predict(xf)
            self.x_pred = xf
            loaded_label = Label(text = 'Prediction for: \n' + self.pred_i.text + '\n\n' + str(prediction))
            pop_data = Popup(title="Prediction results:", content=loaded_label,  size_hint=(None, None), size=(400, 600) )
            pop_data.open()
        except:
            loaded_label = Label(text='Prediction cannot happen!!! \n\nCheck your prediction inputs x \nor the training!!!')
            pop_data = Popup(title="Prediction could not be calculated!!!", content=loaded_label, size_hint=(None, None), size=(400, 400))
            pop_data.open()

    def tf_options(self, instance):
        with open('options.txt') as f:
            txt = f.read()
        pop_Wind = Popup(title="Options", content=Label(text=txt), size_hint=(None,None), size=(400,450))
        pop_Wind.open()

    def save_model(self, instance):
        try:
            self.model.save('models/model'+self.last_training.replace(':',"-"))
            txt = "Model " + self.last_training + "Saved!!! \n\nIt is placed in 'models' folder!!!"
            pop_Wind = Popup(title="Model Saved!!!", content=Label(text=txt), size_hint=(None,None), size=(400,400))
            pop_Wind.open()
        except:
            txt = "Model could not be saved!!! \n\nCheck your training!!!"
            pop_Wind = Popup(title="Model not saved!!!", content=Label(text=txt), size_hint=(None,None), size=(400,400))
            pop_Wind.open()


##################################################################################################
##################################################################################################
###########  initialization


if __name__ == '__main__':
    neuralize_it().run()
