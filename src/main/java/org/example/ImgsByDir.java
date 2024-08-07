package org.example;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class ImgsByDir {

    private static Logger log = LoggerFactory.getLogger(ImgsByDir.class);

    // File representing the folder that you select using a FileChooser
    static final File dir = new File("C:/Users/Admin/Desktop/Учеба/Диплом/test/Natasha24hContr/test2");

    // array of supported extensions (use a List if you prefer)
    static final String[] EXTENSIONS = new String[]{
            "jpg", "png", "bmp" // and other formats you need
    };
    // filter to identify images based on their extensions
    static final FilenameFilter IMAGE_FILTER = new FilenameFilter() {

        @Override
        public boolean accept(final File dir, final String name) {
            for (final String ext : EXTENSIONS) {
                if (name.endsWith("." + ext)) {
                    return (true);
                }
            }
            return (false);
        }
    };

    public static int[] imgNameSplit(String fileName) {

        //the output format:
        //0 - video number
        //1 - cell(trajectory, actually) number
        //2 - frame number

        int[] splittedName = new int[3];
        char[] ch = fileName.toCharArray();
        int dnum = 0;
        List<Character> videoNNameL = new ArrayList<Character>();
        List<Character> cellNumL = new ArrayList<Character>();
        List<Character> timeL = new ArrayList<Character>();
        for (int i = 0; i < ch.length; i++) {
            if (ch[i] == '.') {
                break;
            } else if (ch[i] == '-') {
                dnum++;
            } else {
                switch (dnum) {
                    case (0):
                        videoNNameL.add(ch[i]);
                        break;

                    case (1):
                        cellNumL.add(ch[i]);
                        break;
                    case (2):
                        timeL.add(ch[i]);
                        break;
                }

            }
        }

        String videoName = videoNNameL.stream()
                .map(String::valueOf)
                .collect(Collectors.joining());
        splittedName[0] = Integer.parseInt(videoName);


        String cellNum = cellNumL.stream()
                .map(String::valueOf)
                .collect(Collectors.joining());
        splittedName[1] = Integer.parseInt(cellNum);

        String time = timeL.stream()
                .map(String::valueOf)
                .collect(Collectors.joining());
        splittedName[2] = Integer.parseInt(time);
        return splittedName;
    }


    public static void main(String[] args) throws Exception {


        int height = 51;
        int width = 51;
        int channels = 1; //grayscale
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int batchSize = 128;
        int outputNum = 3;
        int numEpochs = 15;

//        String s = "123-3456-37485.png";
//        int[] arr = imgNameSplit(s);
        //log.info(Arrays.toString(imgNameSplit(s)));

        File cellDefData = new File("C:/Users/Admin/Desktop/Учеба/Диплом/DataV3/DataV2/DataV2/Data/the data/test");
        FileSplit cells = new FileSplit(cellDefData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

//        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels);
//
//        recordReader.initialize(cells);
//        recordReader.setListeners(new LogRecordListener());
//
//        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
//        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
//        scaler.fit(dataIter);
//        dataIter.setPreProcessor(scaler);

        log.info("*******LOADING NN*******");

        File locationToSave = new File("trained_model_RBC.zip");

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        model.getLabels();

        log.info("*******PROCESSING THE DATA*******");

//        float[][] arr = new float[1][1];
//
//        while (dataIter.hasNext()){
//            DataSet next = dataIter.next();
//            INDArray output = model.output(next.getFeatures());
//            arr = output.toFloatMatrix();
//            //log.info(output.toString());
//            //log.info();
//        }
//        Arrays.stream(arr).map(Arrays::toString).forEach(System.out::println);

        if (dir.isDirectory()) { // make sure it's a directory
            for (final File file : dir.listFiles(IMAGE_FILTER)) {
                NativeImageLoader loader = new NativeImageLoader(height, width, channels);
                INDArray image = loader.asMatrix(file);
                DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
                scaler.transform(image);
                INDArray output = model.output(image);
                log.info(file.getName());
                log.info(output.toString());
//                System.out.println("image: " + f.getName());
//                System.out.println(" width : " + img.getWidth());
//                System.out.println(" height: " + img.getHeight());
//                System.out.println(" size  : " + f.length());

            }
            System.out.println(dir.listFiles(IMAGE_FILTER).length);
        }
    }
}


