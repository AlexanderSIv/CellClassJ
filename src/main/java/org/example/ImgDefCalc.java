package org.example;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.*;
import java.util.*;
//import java.util.logging.Logger;
import java.util.logging.FileHandler;
import java.util.logging.SimpleFormatter;
import java.util.stream.Collectors;

public class ImgDefCalc{

    private static Logger log = LoggerFactory.getLogger(ImgDefCalc.class);

    public static class SwingProgressBarExample extends JPanel {

        JProgressBar pbar;

        static final int MY_MINIMUM = 0;

        static final int MY_MAXIMUM = 100;

        public SwingProgressBarExample() {
            // initialize Progress Bar
            pbar = new JProgressBar();
            pbar.setMinimum(MY_MINIMUM);
            pbar.setMaximum(MY_MAXIMUM);
            // add to JPanel
            add(pbar);
        }

        public void updateBar(int newValue) {
            pbar.setValue(newValue);
        }}



    // File representing the folder that you select using a FileChooser

    public static String fileChoose(){
        JFileChooser fc = new JFileChooser();
        fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        int ret = fc.showOpenDialog(null);
        if(ret == JFileChooser.APPROVE_OPTION){
            File file = fc.getSelectedFile();
            String filename = file.getAbsolutePath();
            return filename;
        }
        else {
            return null;
        }
    }
    static final String filechoose = fileChoose().toString();
    //    static final File dir = new File("C:/Users/Admin/Desktop/Учеба/Диплом/Natasha24hPTX/Natasha24hPTX1/test");
    static final File dir = new File(filechoose);

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

    public static double[][] cunningSort(double[][] imgInfo) {  //images are read in wrong order. let's fix it
        double[][] infoSorted = new double[imgInfo.length][imgInfo[0].length - 1];
        int cellBegin = 0;
        int cellEnd = 0;
        double tempRow = 0;
        int cellcount = 1;
        infoSorted[0][0] = cellcount;
        for (int i = 1; i < infoSorted.length; i++) {
            if (imgInfo[i][0] == imgInfo[i - 1][0] && imgInfo[i][1] == imgInfo[i - 1][1]) {
                cellEnd++;
                infoSorted[i][0] = cellcount;
            } else {
                cellcount++;
                infoSorted[i][0] = cellcount;
            }
        }
        for (int i = 0; i < infoSorted.length; i++) {
            for (int j = 1; j < infoSorted[0].length; j++) {
                infoSorted[i][j] = imgInfo[i][j + 1];
            }
        }
        for (int i = 0; i < infoSorted.length; i++) {
            for (int j = 0; j < infoSorted.length; j++) {
                if (infoSorted[i][0] == infoSorted[j][0] && j > i && infoSorted[j][1] < infoSorted[i][1]) {
                    for (int k = 0; k < infoSorted[0].length; k++) {
                        tempRow = infoSorted[i][k];
                        infoSorted[i][k] = infoSorted[j][k];
                        infoSorted[j][k] = tempRow;
                    }
                }
            }
        }
        return infoSorted;
    }

    public static double[][] getShapeDiff(double[][] imgInfo) {  //fix img order and extract shape diff
        double[][] infoSorted = new double[imgInfo.length][imgInfo[0].length - 1];
        double[][] infoDiff = new double[imgInfo.length][3];
        int cellBegin = 0;
        int cellEnd = 0;
        double tempRow = 0;
        int cellcount = 1;
        infoSorted[0][0] = cellcount;
        for (int i = 1; i < infoSorted.length; i++) {
            if (imgInfo[i][0] == imgInfo[i - 1][0] && imgInfo[i][1] == imgInfo[i - 1][1]) {
                cellEnd++;
                infoSorted[i][0] = cellcount;
            } else {
                cellcount++;
                infoSorted[i][0] = cellcount;
            }
        }
        for (int i = 0; i < infoSorted.length; i++) {
            for (int j = 1; j < infoSorted[0].length; j++) {
                infoSorted[i][j] = imgInfo[i][j + 1];
            }
        }
        for (int i = 0; i < infoSorted.length; i++) {
            for (int j = 0; j < infoSorted.length; j++) {
                if (infoSorted[i][0] == infoSorted[j][0] && j > i && infoSorted[j][1] < infoSorted[i][1]) {
                    for (int k = 0; k < infoSorted[0].length; k++) {
                        tempRow = infoSorted[i][k];
                        infoSorted[i][k] = infoSorted[j][k];
                        infoSorted[j][k] = tempRow;
                    }
                }
            }
        }
        int cellOrigin = 0;
        int currentCellNum = 1;
        infoDiff[0][0] = infoSorted[0][0];
        infoDiff[0][1] = infoSorted[0][1];
        infoDiff[0][2] = 0;
        for (int i = 1; i < infoSorted.length; i++) {
            infoDiff[i][0] = infoSorted[i][0];
            infoDiff[i][1] = infoSorted[i][1];
            if(infoSorted[i][0] == infoSorted[i-1][0]){
                double d = 0;
                for(int j = 2; j< infoSorted[0].length; j++){
                    d += (infoSorted[i][j]-infoSorted[i-1][j])*(infoSorted[i][j]-infoSorted[i-1][j]);
                }
                d = Math.sqrt(d);
                infoDiff[i][2] = d;
            } else {
                infoDiff[i][2] = 0;
            }
        }
        return infoDiff;
    }

    public static void main(String[] args) throws Exception {

        final SwingProgressBarExample it = new SwingProgressBarExample();

        JFrame frame = new JFrame("Please wait");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setContentPane(it);
        //frame.pack();
        frame.setSize(300, 200);
        frame.setVisible(true);

//        int height = 51;
//        int width = 51;
//        int channels = 1; //grayscale
//        int rngseed = 123;
//        Random randNumGen = new Random(rngseed);
//        int outputNum = 3;

        //import settings from config file
        Properties props = new Properties();
        String fileName = "calc_settings.config";
        FileInputStream fis = new FileInputStream(fileName);
        props.load(fis);
        int channels = Integer.valueOf(props.getProperty("channels"));
        int height = Integer.valueOf(props.getProperty("height"));
        int width = Integer.valueOf(props.getProperty("width"));
        int outputNum = Integer.valueOf(props.getProperty("outputNum"));


        log.info("*******LOADING NN*******");

        File locationToSave = new File("trained_model_RBC.zip");

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        model.getLabels();

        log.info("*******PROCESSING THE DATA*******");


        if (dir.isDirectory()) { // make sure it's a directory
            int numberOfImages = dir.listFiles(IMAGE_FILTER).length;
            double[][] imgInfo = new double[numberOfImages][3 + outputNum];
            // format: video number & cell number & coordinates of the pictures
            int i = 0;
            NativeImageLoader loader = new NativeImageLoader(height, width, channels);
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            int[] imgname = new int[3];
            double[] coord = new double[outputNum];
            for (final File file : dir.listFiles(IMAGE_FILTER)) {
                final int percent = Math.round(100*i/numberOfImages);
                INDArray image = loader.asMatrix(file);
                scaler.transform(image);
                INDArray output = model.output(image);
                imgname = imgNameSplit(file.getName());
                imgInfo[i][0] = imgname[0];
                imgInfo[i][1] = imgname[1];
                imgInfo[i][2] = imgname[2];
                coord = output.toDoubleVector();
                for (int j = 0; j < outputNum; j++) {
                    imgInfo[i][j + 3] = coord[j];
                }
                i++;
                it.updateBar(percent);
                System.out.println(i + " of " + numberOfImages + " are done");
            }


            //let's print our raw data, just in case
            FileWriter fileWriterCoord = new FileWriter("imgCoord.txt");
            PrintWriter coordWriter = new PrintWriter(fileWriterCoord);
            for (int k = 0; k < imgInfo.length; k++) {
                for (int j = 0; j < imgInfo[0].length; j++) {
                    coordWriter.print(imgInfo[k][j] + " ");
                }
                coordWriter.println();
            }
            coordWriter.close();

            //coordinate sorted data, for other usages
            FileWriter fileWriterCell = new FileWriter("cellCoord.txt");
            double[][] cellCoord = cunningSort(imgInfo);
            PrintWriter cellWriter = new PrintWriter(fileWriterCell);
            for (int k = 0; k < cellCoord.length; k++) {
                for (int j = 0; j < cellCoord[0].length; j++) {
                    cellWriter.print(cellCoord[k][j] + " ");
                }
                cellWriter.println();
            }
            cellWriter.close();

            // there is our main shape output file
            FileWriter fileWriterInfo = new FileWriter("imgInfo.txt");
            imgInfo = getShapeDiff(imgInfo);
            PrintWriter infoWriter = new PrintWriter(fileWriterInfo);
            for (int k = 0; k < imgInfo.length; k++) {
                for (int j = 0; j < imgInfo[0].length; j++) {
                    infoWriter.print(imgInfo[k][j] + " ");
                }
                infoWriter.println();
            }

            infoWriter.close();
            System.out.println("************Proccess finished normal************");
            System.exit(0);
        }
    }
}



