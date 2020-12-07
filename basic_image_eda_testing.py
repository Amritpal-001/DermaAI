from basic_image_eda import BasicImageEDA

if __name__ == "__main__":  # for multiprocessing
    data_dir = '/home/amritpal/PycharmProjects/100-days-of-code/100_days_of_code/Skin_lesions_Classification-master/data/Train/actinic keratosis'
    extensions = ['png', 'jpg', 'jpeg']
    threads = 0
    dimension_plot = True
    channel_hist = True
    nonzero = False
    hw_division_factor = 1.0

    BasicImageEDA.explore(data_dir, extensions, threads, dimension_plot, channel_hist, nonzero, hw_division_factor)
    #BasicImageEDA.explore(data_dir)


'''number of images                         |  16

dtype                                    |  uint8
channels                                 |  [3]
extensions                               |  ['jpg']

min height                               |  450
max height                               |  768
mean height                              |  489.75
median height                            |  450

min width                                |  600
max width                                |  1024
mean width                               |  653.0
median width                             |  600

mean height/width ratio                  |  0.75
median height/width ratio                |  0.75
recommended input size(by mean)          |  [488 656] (h x w, multiples of 8)
recommended input size(by mean)          |  [496 656] (h x w, multiples of 16)
recommended input size(by mean)          |  [480 640] (h x w, multiples of 32)

channel mean(0~1)                        |  [0.74523723 0.53253925 0.5621598 ]
channel std(0~1)                         |  [0.12346011 0.11388277 0.13660207]'''