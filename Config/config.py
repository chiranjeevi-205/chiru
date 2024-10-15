class Config():
    # all about dataset generation
    base_path = "./Dataset/"

    # train dataset prepare
    train_img_paths = 'Birds_train/train_image_paths.txt'
    train_triplet_indexes = 'Birds_train/train_triplet_index.txt'
    train_root_dir ='Birds_train/'
    n_train_sample = 10000


    # test dataset prepare
    test_img_paths = 'Birds_test/test_image_paths.txt'
    test_triplet_indexes = 'Birds_test/test_triplet_index.txt'
    test_root_dir ='Birds_test/'
    n_test_sample = 5000

    # training parameter
    train_batch_size = 8
    test_batch_size = 8
    n_epochs = 30
    momentum = 0.5
    lr = 0.001
    l2R = 0.0001 # L2 regularization
    margin = 1.0

    # for inference
    embb_data = './embb_data/'
    model_weights = './runs_10k_99.78_FA_l2_0.0001/TripletNet/model_best.pth'
    yolo_weights = './yolo_weights/best.pt'
    embb_space = './encodings/encodings.pkl'
    input_video = './test.mp4'
    output_infer = './infer_output/output.mp4'

        
