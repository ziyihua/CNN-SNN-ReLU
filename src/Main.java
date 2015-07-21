import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLNumericArray;

import java.io.*;
import java.util.ArrayList;
import java.util.Map;

/**
 * Created by ziyihua on 17/07/15.
 */
public class Main {

    public static void main(String[] args) throws IOException {
        MatFileReader m = new MatFileReader("matnativedouble.mat");
        double[][] mlArrayDouble = ((MLDouble) m.getMLArray("arr")).getArray();
        System.out.println(mlArrayDouble[0][0]);


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
        architecture[2][1] = "6";
        architecture[2][3] = "12";

        Structure.network convnet = new Structure.network();
        convnet = SetUp.SetUp(architecture);
        convnet.first_layer_dropout=0.0f;


        //import images and labels
        int[] label_t;
        label_t = ImportFile.getLabel("train-labels-idx1-ubyte");
        float[][][] image_t;
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
        float alpha = 1;
        int batchsize = 50;
        int numepochs = 1;
        int learn_bias = 0;
        float dropout = 0.0f;

        convnet = CNNtrain.CNNTrain(architecture,convnet,alpha,numepochs,batchsize,dropout,learn_bias,label_t_new,image_t);

        /*float[] loss = new float[convnet.rL.length];
        for (int i = 0; i < convnet.rL.length; i++) {
            loss[i]=convnet.rL[i];
        }
        int[] indx = new int[loss.length];
        for (int i = 0; i < loss.length; i++) {
            indx[i] = i;
        }*/

        int[] test_y;
        test_y = ImportFile.getLabel("t10k-labels-idx1-ubyte");
        float[][][] test_x;
        test_x = ImportFile.getImage("t10k-images-idx3-ubyte");

        int[][] test_y_new = new int[10][test_y.length];
        for (int i = 0; i < test_y.length ; i++) {
            for (int j = 0; j < 10; j++) {
                if (test_y[i]==j){
                    test_y_new[j][i]=1;
                }else {
                    test_y_new[j][i]=0;
                }
            }
        }

        float rate = CNNtest.CNNtest(architecture,convnet,test_x,test_y);
        convnet.acc = rate;
        System.out.println(rate);

        for (int i = 0; i < convnet.layers.get(1).k.size(); i++) {
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < 5; k++) {
                    System.out.println(convnet.layers.get(1).k.get(i)[j][k]);
                }
            }
            System.out.println("lalala");
        }

        /*try
        {
            FileOutputStream fileOut =
                    new FileOutputStream("/tmp/cnn.ser");
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(convnet);
            out.close();
            fileOut.close();
            System.out.printf("Serialized data is saved in /tmp/cnn.ser");
        }catch(IOException i)
        {
            i.printStackTrace();
        }

        Plot2DPanel plot = new Plot2DPanel();
        plot.addLegend("SOUTH");

        // add a line plot to the PlotPanel
        plot.addLinePlot("Squared Loss", indx, loss);

        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame("a plot panel");
        frame.setSize(600, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);

        Structure.network convnet = null;
        try {
            FileInputStream fileIn = new FileInputStream("cnn_9809.ser");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            convnet = (Structure.network) in.readObject();
            in.close();
            fileIn.close();
        } catch (IOException i) {
            i.printStackTrace();
            return;
        } catch (ClassNotFoundException c) {
            System.out.println("network not found");
            c.printStackTrace();
            return;
        }
        System.out.println("Deserialized network...");*/

        //convnet = Convlifsim.Convlifsim(convnet,test_x,test_y_new,test_y,0.000f,1.0f,0.001f,0.020f,0.001f,1000);

        /*double[] loss_e = new double[e.rL.length];
        for (int i = 0; i < e.rL.length; i++) {
            loss_e[i]=e.rL[i];
        }
        double[] indx_e = new double[loss_e.length];
        for (int i = 0; i < loss_e.length; i++) {
            indx_e[i] = i;
        }
        Plot2DPanel plot_e = new Plot2DPanel();
        plot_e.addLegend("SOUTH");

        // add a line plot to the PlotPanel
        plot_e.addLinePlot("Squared Loss", indx_e, loss_e);

        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame_e = new JFrame("a plot panel");
        frame_e.setSize(600, 600);
        frame_e.setContentPane(plot_e);
        frame_e.setVisible(true);*/
    }
}
