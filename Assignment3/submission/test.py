import os , csv, shutil
import keras
import tensorflow as tf
from classification import generate_label_folders,data_generator, count_files
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score


img_width, img_height = 500, 580
def gen_ic_dict():
    iclabel = {}
    with open("testPatient/test_Labels.csv", 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            iclabel[row['IC']] = int(row['Label'])
    return iclabel


def evaluate(dataFolder): 
    test_dir = "test_data"
    try:
        shutil.rmtree(test_dir)
    except:
        pass
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_rsn_dir = os.path.join(test_dir, "RSN")
    test_noise_dir = os.path.join(test_dir, "Noise")
    dirs = [ test_rsn_dir, test_noise_dir]
    for dir in dirs:
        try:
            shutil.rmtree(dir)   
        except:
            pass
        os.makedirs(dir)
    
    generate_label_folders(dataFolder, "1", test_rsn_dir, test_noise_dir, gen_ic_dict())
    

    data_size = count_files(test_rsn_dir) + count_files(test_noise_dir)
    datagen = ImageDataGenerator(rescale=1. / 255)
    data_gen = datagen.flow_from_directory("test_data",target_size=(img_width, img_height), batch_size=1, shuffle = False, class_mode='binary')
    
    # # val_ds = data_generator(test_dir,batch_size =10, stage="val")
    model = keras.models.load_model('model/save_at_18_0.436789.keras')
    result = model.predict(data_gen)
    result[result > 0.5] = 1
    result[result <= 0.5] = 0
    result = result.squeeze()
    labels = zip(data_gen.filenames[:len(result)],result)
    labels = sorted(list(labels), key=lambda x: int(x[0].split("_")[1]))
    preds = [l[1] for l in labels]
    true_labels = []
    with open("testPatient/test_Labels.csv" , "r") as inp :
        header = inp.readline()
        for line in inp:
            line = line.strip()
            ic, label = line.split(",")
            if label == "0":
                true_labels.append(0)
            else:
                true_labels.append(1)
    conf_mat = confusion_matrix(true_labels, preds)
    print(conf_mat)
    preci = conf_mat[1][1]/ (conf_mat[1][1]+conf_mat[0][1])
    accuracy = (conf_mat[1][1] + conf_mat[0][0])/(conf_mat[1][1] + conf_mat[0][0]+ conf_mat[0][1] + conf_mat[1][0])
    sensitivity = conf_mat[1][1]/(conf_mat[1][1]+conf_mat[0][1])
    specificity = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[1][0])
    print(preci, accuracy, sensitivity, specificity)

    with open("Metrics.csv", "w", encoding="utf-8") as out:
        out.write(",".join(["Accuracy", f"{accuracy*100:.2f}%"])+"\n")
        out.write(",".join(["Precision", f"{preci*100:.2f}%"])+"\n")
        out.write(",".join(["Sensitivity", f"{sensitivity*100:.2f}%"])+"\n")
        out.write(",".join(["Specificity", f"{specificity*100:.2f}%"])+"\n")
    
    labels = sorted(list(labels), key=lambda x: int(x[0].split("_")[1]))
    with open("Results.csv", "w", encoding="utf-8") as out:
        out.write(",".join(["IC_Number", "Label"])+"\n")
        for f,l in labels:
            f = f.split("_")[1]
            out.write(",".join([f,str(int(l))])+"\n")


evaluate("testPatient/test_Data")
    
