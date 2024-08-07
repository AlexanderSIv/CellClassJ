package org.example;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.CollectScoresIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.FileInputStream;
import java.util.Properties;
import java.util.Random;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class ImgTrainModel extends javax.swing.JFrame{

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
    private static Logger log = Logger.getLogger(String.valueOf(ImgTrainModel.class));
    static FileHandler fh;

    public static void main(String[] args) throws Exception {
        //building simple UI
        JFrame frame = new JFrame("Model training");
        frame.setLayout(new GridBagLayout());
        frame.setSize(300, 200);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        fh = new FileHandler("Train.log");
        log.addHandler(fh);
        SimpleFormatter formatter = new SimpleFormatter();
        fh.setFormatter(formatter);


        JLabel prepData = new JLabel("Preparing the data...");
        frame.add(prepData);


        //load img and model properties from config file
        Properties prop = new Properties();
        String fileName = "parameters.config";
        FileInputStream fis = new FileInputStream(fileName);
        prop.load(fis);
        int channels = Integer.valueOf(prop.getProperty("channels"));
        int height = Integer.valueOf(prop.getProperty("height"));
        int width = Integer.valueOf(prop.getProperty("width"));
        int batchSize = Integer.valueOf(prop.getProperty("batchSize"));
        int outputNum = Integer.valueOf(prop.getProperty("outputNum"));
        int numEpochs = Integer.valueOf(prop.getProperty("numEpochs"));
        double l2val = Double.valueOf(prop.getProperty("l2val"));
        double learningRate = Double.valueOf(prop.getProperty("learningRate"));
        double momentum = Double.valueOf(prop.getProperty("momentum"));
        int rngseed = Integer.valueOf(prop.getProperty("rngseed"));

//        int height = 51;
//        int width = 51;
//        int channels = 1; //grayscale
//        int rngseed = 1234;
        Random randNumGen = new Random(rngseed);
//        int batchSize = 128;
//        int outputNum = 3;
//        int numEpochs = 10;
//        double l2val = 0.005;
//        double learningRate = 0.005;
//        double momentum = 0.9;


        //data prop
        File trainData = new File(filechoose);
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);


        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        recordReader.initialize(train);
        // recordReader.setListeners(new LogRecordListener());

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        frame.remove(prepData);
        JLabel buildNet = new JLabel("Building NN...");
        frame.add(buildNet);
        frame.invalidate();
        frame.validate();
        frame.repaint();
        //building NN

        log.info("*******BUILDING NN*******");

        //this is simple NN sample, i don't believe it'll work

//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(rngseed)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
//                .updater(new Nesterovs(0.006,0.9))
//                .l2(1e-4)
//                .list()
//                .layer(0, new DenseLayer.Builder()
//                        .nIn(height*width)
//                        .nOut(100)
//                        .activation(Activation.RELU)
//                        .weightInit(WeightInit.XAVIER)
//                        .build())
//                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nIn(100)   //same as nOut early
//                        .nOut(outputNum)
//                        .activation(Activation.SOFTMAX)
//                        .weightInit(WeightInit.XAVIER)
//                        .build())
//                .setInputType(InputType.convolutional(height,width,channels))
//                .build();

        //A big CNN

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngseed)
                .l2(l2val)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learningRate,momentum))
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{7,7}, new int[]{1,1}, new int[]{0,0})
                        .nIn(channels)
                        .nOut(10)
                        .biasInit(0)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, new int[]{0,0})
                        .nOut(20)
                        .biasInit(0)
                        .build())
                .layer(3,new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).build())
                .layer(4, new DenseLayer.Builder().nOut(200).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

//        model.setListeners(new ScoreIterationListener(10));
        CollectScoresIterationListener listener = new CollectScoresIterationListener(1);
        model.setListeners(listener);

        frame.remove(buildNet);
        JLabel trNet = new JLabel("Training the NN...");
        frame.add(trNet);
        frame.invalidate();
        frame.validate();
        frame.repaint();

        log.info("*******TRAINING********");
        for (int i=0;i<numEpochs;i++){
            JLabel eps = new JLabel(" "+String.valueOf(i)+" epochs of "+String.valueOf(numEpochs)+" are done!");
            frame.add(eps);
            frame.invalidate();
            frame.validate();
            frame.repaint();
            model.fit(dataIter);
            frame.remove(eps);
            frame.invalidate();
            frame.validate();
            frame.repaint();
        }
        listener.exportScores(new File("scores.log"));

        frame.remove(trNet);
        JLabel svn = new JLabel("Saving trained model...");
        frame.add(svn);
        frame.invalidate();
        frame.validate();
        frame.repaint();
        //saving the model

        log.info("*********SAVING TRAINED MODEL*******");

        File locationToSave = new File("trained_model_RBC.zip");
        boolean saveUpdater = false;

        ModelSerializer.writeModel(model, locationToSave, saveUpdater);

        System.exit(0);

    }

}

