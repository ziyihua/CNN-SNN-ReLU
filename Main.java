import org.math.plot.Plot2DPanel;

import javax.swing.*;
import java.io.IOException;

/**
 * Created by ziyihua on 17/07/15.
 */
public class Main {

    public static void main(String[] args) throws IOException{


        //Layers of CNN
        //row 0 is type
        String[][] architecture = new String[3][5];
        architecture[0][0] = "i";
        architecture[0][1] = "c";
        architecture[0][2] = "s";
        architecture[0][3] = "c";
        architecture[0][4] = "s";
        //row 1 is kernelsize for convolutional layer and scale for subsampling layer
        architecture[1][1] = "5";
        architecture[1][2] = "2";
        architecture[1][3] = "5";
        architecture[1][4] = "2";
        //row 2 is number of output maps for convolutional layer
        architecture[2][1] = "16";
        architecture[2][3] = "16";

        Structure.network convnet = new Structure.network();
        convnet = SetUp.SetUp(architecture);
        convnet.first_layer_dropout=0.0;


        //import images and labels
        int[] label_t;
        label_t = ImportFile.getLabel("train-labels-idx1-ubyte");
        double[][][] image_t;
        image_t = ImportFile.getImage("train-images-idx3-ubyte");


        int[][] label_t_new = new int[10][label_t.length];
        for (int i = 0; i < label_t.length ; i++) {
            for (int j = 0; j < 10; j++) {
                if (label_t[i]==j){
                    label_t_new[j][i]=1;
                }else {
                    label_t_new[j][i]=0;
                }
            }
        }


        //parameters
        double alpha = 1;
        int batchsize = 50;
        int numepochs = 1;
        int learn_bias = 0;
        double dropout = 0.0;

        convnet = CNNtrain.CNNTrain(architecture,convnet,alpha,numepochs,batchsize,dropout,learn_bias,label_t_new,image_t);

        double[] loss = new double[convnet.rL.length];
        for (int i = 0; i < convnet.rL.length; i++) {
            loss[i]=convnet.rL[i];
        }
        double[] indx = new double[loss.length];
        for (int i = 0; i < loss.length; i++) {
            indx[i] = i;
        }

        int[] test_y;
        test_y = ImportFile.getLabel("t10k-labels-idx1-ubyte");
        double[][][] test_x;
        test_x = ImportFile.getImage("t10k-images-idx3-ubyte");

        double rate = CNNtest.CNNtest(architecture,convnet,test_x,test_y);
        System.out.println(rate);

        Plot2DPanel plot = new Plot2DPanel();
        plot.addLegend("SOUTH");

        // add a line plot to the PlotPanel
        plot.addLinePlot("Squared Loss", indx, loss);

        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame("a plot panel");
        frame.setSize(600, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);
    }
}
