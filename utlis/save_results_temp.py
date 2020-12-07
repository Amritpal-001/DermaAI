
from init_variables import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import time


def print_results(History):
    print(History.history["accuracy"])
    print(History.history['val_accuracy'])
    print(History.history['loss'])
    print(History.history['val_loss'])

def plot_results(history,save_location ,  save_graph = True ):

    plt.plot(history.history["accuracy"])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Inception V3")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
    plt.show()
    if save_graph == True:
        plt.savefig(save_location + '/books_read.png')
        print('saved graph')


def save_to_fixed_column(sheet , variable_to_save , starting_row , column_number):
    for a in range(0 , len(variable_to_save)):
        sheet.write(column_number , starting_row + a, variable_to_save[a])

def add_variables_as_csv(sheet , variable ,save_location , save_graph = True  ):
    sheet.write(0, 0, 'accuracy')
    save_to_fixed_column(sheet, variable, starting_row, column_number)
    print('saved graph')

def save_results_as_excel(History, save_location , save_graph = True ):
    from xlwt import Workbook
    wb = Workbook()
    sheet = wb.add_sheet('Sheet 1')
    sheet.write(0, 0, 'accuracy')
    sheet.write(1, 0, 'val_accuracy')
    sheet.write(2, 0, 'loss')
    sheet.write(3, 0, 'val_loss')

    accuracy = History.history["accuracy"]
    val_accuracy = History.history['val_accuracy']
    loss = History.history['loss']
    val_loss = History.history['val_loss']

    starting_row = 1
    save_to_fixed_column(sheet, accuracy, starting_row, 0)
    save_to_fixed_column(sheet, val_accuracy, starting_row, 1)
    save_to_fixed_column(sheet, loss, starting_row, 2)
    save_to_fixed_column(sheet, val_loss, starting_row, 3)

    wb.save(save_location + '/results.xls')
    print('saved results to xls')



#reader = pd.read_csv(r'/home/amritpal/PycharmProjects/100-days-of-code/Hackathon/working_version_minimal/utlis/results.csv')
#print(reader)
#reader.to_csv('amrit_test.csv')

def save_results_as_log_file_csv(History, save_location):
    till_date_logs = pd.read_csv( save_location + '/log_file_results_only.csv' ,  index_col=False)

    #print(till_date_logs)
    #print('shape of dataframe = ', till_date_logs.shape)
    #len_of_columns_in_log_file = till_date_logs.shape[0]
    #print('len_of_columns_in_log_file = ', len_of_columns_in_log_file)

    accuracy = History.history["accuracy"]
    val_accuracy = History.history['val_accuracy']
    loss = History.history['loss']
    val_loss = History.history['val_loss']

    New_entry = pd.DataFrame({"accuracy": accuracy,
                              "val_accuracy": val_accuracy,
                              "loss": loss,
                              "val_loss": val_loss
                              })

    empty_dataframe = pd.DataFrame({"accuracy": [0, 0],
                                    "val_accuracy": [0, 0],
                                    "loss": [0, 0],
                                    "val_loss": [0, 0],
                                    })

    New_logs = till_date_logs.append(New_entry, ignore_index=True)
    print(New_logs)
    New_logs = New_logs.append(empty_dataframe, ignore_index=True)

    New_logs.to_csv(save_location + '/log_file_results_only.csv', index=False)
    print('saved results to xls')


def save_results_as_log_file_csv_updated(History, model , epochs , learning_rate , image_size , batch_size , save_location ):
    till_date_logs = pd.read_csv( save_location + '/log_file_results_only.csv' ,  index_col=False)



    #print('shape of dataframe = ', till_date_logs.shape)
    #len_of_columns_in_log_file = till_date_logs.shape[0]
    #print('len_of_columns_in_log_file = ', len_of_columns_in_log_file)

    local_time = time.ctime(time.time())
    # model parameters size
    model = model
    #
    #
    epochs = epochs
    learning_rate = learning_rate
    image_size = image_size
    batch_size = batch_size


    accuracy = History.history["accuracy"]
    val_accuracy = History.history['val_accuracy']
    loss = History.history['loss']
    val_loss = History.history['val_loss']

    New_entry = pd.DataFrame({"accuracy": accuracy,
                              "val_accuracy": val_accuracy,
                              "loss": loss,
                              "val_loss": val_loss
                              })

    empty_dataframe = pd.DataFrame({"accuracy": [0, 0],
                                    "val_accuracy": [0, 0],
                                    "loss": [0, 0],
                                    "val_loss": [0, 0],
                                    })

    New_logs = till_date_logs.append(New_entry, ignore_index=True)
    print(New_logs)
    New_logs = New_logs.append(empty_dataframe, ignore_index=True)

    New_logs.to_csv(save_location + '/log_file_results_only.csv', index=False)
    print('saved results to xls')


def gen_model_name(model_name  , image_size , black_and_white):
    if black_and_white == True:
        mode = 'L'
    else:
        mode = 'RGB'
    name = model_name + '_' + mode + '_' + str(image_size)
    print(name)
    return(name)


def saving_name(model_name , History , round_off_till ):
    current_time = time.localtime()
    #model_name
    name = model_name + '__' + str(round(History.history['val_accuracy'][-1] , round_off_till)) + \
           '__' + str(round(History.history['val_loss'][-1] , round_off_till)) +  '__' + str(current_time.tm_mon)  \
           + '-' +  str(current_time.tm_mday)  + '_' + str(current_time.tm_hour) + ':' + str(current_time.tm_min)
    print(name)
    return(name)

#generate_name('model_name ')