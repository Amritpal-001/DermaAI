
from init_variables import *
import matplotlib.pyplot as plt

import os
from convert_to_tensorflow_lite import *


def print_results(History):
    print(History.history("accuracy"])
    print(History.history('val_accuracy'])
    print(History.history('loss'])
    print(History.history('val_loss'])

def plot_results(History, save_graph = True , save_location):
    plt.plot(History.history("accuracy"])
    plt.plot(History.history('val_accuracy'])
    plt.plot(History.history('loss'])
    plt.plot(History.history('val_loss'])
    plt.title("Inception V3")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(("Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
    plt.show()
    if save_graph == True:
        plt.savefig(save_location + '/books_read.png')

    print('saved graph')

def save_to_fixed_column(sheet , variable_to_save , starting_row , column_number):
    for a in range(0 , len(variable_to_save)):
        sheet.write(column_number , starting_row + a, variable_to_save[a])

'''def add_variables_as_csv(sheet , variable , save_graph = True , save_location):
    sheet.write(0, 0, 'accuracy')
    save_to_fixed_column(sheet, variable, starting_row, column_number)
    print('saved graph')'''

def save_results_as_csv(History, save_graph = True , save_location):
    from xlwt import Workbook
    wb = Workbook()
    sheet = wb.add_sheet('Sheet 1')
    sheet.write(0, 0, 'accuracy')
    sheet.write(0, 1, 'val_accuracy')
    sheet.write(0, 2, 'loss')
    sheet.write(0, 3, 'val_loss')

    accuracy = History.history["accuracy"]
    val_accuracy = History.history['val_accuracy']
    loss = History.history['loss']
    val_loss = History.history['val_loss']

    save_to_fixed_column(sheet, variable_to_save, starting_row, column_number)
    save_to_fixed_column(sheet, variable_to_save, starting_row, column_number)
    save_to_fixed_column(sheet, variable_to_save, starting_row, column_number)
    save_to_fixed_column(sheet, variable_to_save, starting_row, column_number)

    wb.save(save_location + '/results.xls')

    print('saved graph')


def make_model_folder(model_name):
    model_direc = './' + model_name
    os.mkdir(model_direc)
    global model_directory  = model_direc


def model_allocated_name(model_architecutre ):
    # based on model name , image_size , result validation accuracy , time


def zip_model_file(folder_location , download = True)
        # zip the model folder with

        if download == True:
            # downloads it

        # Code_pending
        #
        #

def simple_model_traning_details_to_save(model, seed_value, numpy_seed_value, tensorflow_seed_value,
                                              model_allocated_name, model_architecture,
                                              learning_rate, Initial_DL_rate, Final_DL_rate, epoch, optimizer, metrics,
                                              image_type, img_size, batch_size, data_directory, data_size, data_classes,
                                              accuracy, val_accuracy, loss, val_loss, topKloss):

                model = model

                model_detials_variable = []

                ####### results
                accuracy = accuracy
                val_accuracy = val_accuracy
                loss = loss
                val_loss = val_loss
                topKloss = topKloss

                final_accuracy = accuracy[-1]
                final_val_accuracy = val_accuracy[-1]
                final_loss = loss[-1]
                final_val_loss = val_loss[-1]
                final_topKloss = topKloss[-1]

                model_detials_variable.append(accuracy)  # accuracy
                model_detials_variable.append(final_accuracy)  # final_accuracy

                model_detials_variable.append(val_accuracy)  # val_accuracy
                model_detials_variable.append(final_val_accuracy)  # final_val_accuracy

                model_detials_variable.append(loss)  # loss
                model_detials_variable.append(final_loss)  # final_loss

                model_detials_variable.append(val_loss)  # val_loss
                model_detials_variable.append(final_val_loss)  # final_val_loss

                model_detials_variable.append(topKloss)  # topKloss
                model_detials_variable.append(final_topKloss)  # final_topKloss

                ################### seed setting
                model_detials_variable.append(seed_value)  # seed_value
                model_detials_variable.append(numpy_seed_value)  # numpy_seed_value
                model_detials_variable.append(tensorflow_seed_value)  # tensorflow_seed_value

                saving_directory = './saved_results'
                Saving_accuracy_threshold = 70
                saved_plot_image_name =

                if save_model = True:
                    save

                if zip_model_condition = True:
                    zip_model_file

                return (model_detials_variable)


def model_traning_details_to_save(model , seed_value , numpy_seed_value , tensorflow_seed_value , model_allocated_name , model_architecture ,
                                  learning_rate ,Initial_DL_rate , Final_DL_rate ,epoch , optimizer  , metrics ,
                                  image_type , img_size , batch_size , data_directory , data_size , data_classes ,
                                  accuracy , val_accuracy , loss , val_loss , topKloss ):


        model = model

        model_detials_variable = []

        ####### results
        accuracy = accuracy
        val_accuracy =  val_accuracy
        loss = loss
        val_loss = val_loss
        topKloss =  topKloss

        final_accuracy = accuracy[-1]
        final_val_accuracy = val_accuracy[-1]
        final_loss = loss[-1]
        final_val_loss = val_loss[-1]
        final_topKloss = topKloss[-1]

        model_detials_variable.append(accuracy)     #accuracy
        model_detials_variable.append(final_accuracy)   #final_accuracy

        model_detials_variable.append(val_accuracy)  #val_accuracy
        model_detials_variable.append(final_val_accuracy)    #final_val_accuracy

        model_detials_variable.append(loss)                   #loss
        model_detials_variable.append(final_loss)             #final_loss

        model_detials_variable.append(val_loss)               #val_loss
        model_detials_variable.append(final_val_loss)         #final_val_loss

        model_detials_variable.append(topKloss)               #topKloss
        model_detials_variable.append(final_topKloss)         #final_topKloss

        ################## model architecutre
        model_detials_variable.append(model_allocated_name)    #model_allocated_name
        model_detials_variable.append(model_architecture)      #model_architecture
        model_parameters_size =
        last_layer_activation =
        last_layer_classes =
        model_detials_variable.append(model_parameters_size)   #model_parameters_size
        model_detials_variable.append(last_layer_activation)   #last_layer_activation
        model_detials_variable.append(last_layer_classes)      #last_layer_classes

        ################## training details
        model_detials_variable.append(learning_rate)    #learning_rate
        model_detials_variable.append(Initial_DL_rate)  #Initial_DL_rate
        model_detials_variable.append(Final_DL_rate )   #Final_DL_rate
        model_detials_variable.append(epoch)            #epoch
        model_detials_variable.append(optimizer)        #optimizer
        model_detials_variable.append(metrics)          #metrics
        loss_functions =
        training_start_time =
        total_training_time =
        model_detials_variable.append(loss_functions)
        model_detials_variable.append(training_start_time)
        model_detials_variable.append(total_training_time)

        ################### image_preprocessing features
        model_detials_variable.append(image_type)  #image_type
        model_detials_variable.append(img_size)  #img_size
        model_detials_variable.append(batch_size)  #batch_size
        model_detials_variable.append(data_size)  #data_size
        model_detials_variable.append(data_classes) #data_classes
        model_detials_variable.append(data_directory)  #data_directory


        ################### seed setting
        model_detials_variable.append(seed_value) # seed_value
        model_detials_variable.append(numpy_seed_value)   #numpy_seed_value
        model_detials_variable.append(tensorflow_seed_value) #tensorflow_seed_value

        saving_directory = './saved_results'
        Saving_accuracy_threshold = 70
        saved_plot_image_name =

        if save_model = True:
            save

        if zip_model_condition = True:
            zip_model_file


        return(model_detials_variable)


#def model_weights_save(model):




def model_detials_variable_to_Log_file(model_detials_variable  , log_excel):







'''
def saving_condition():
    # Code_pending
    #
    #
    if variable >= threshold:
        return (True)
    if variable < threshold:
        return (False)


def custom_save_model(model_name):
    save_training_resuls(model_name)

    if saving_condition() == True:
        # save_model code
        # Code_pending
        #
        #

        #
        #
        # convert_to_tensorflow_lite model
        # Code_pending
        #
        #
        convert_to_tensorflow_lite(model_name)
        
'''




