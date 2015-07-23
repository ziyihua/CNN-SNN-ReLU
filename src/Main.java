import com.jmatio.io.MatFileReader;
import com.jmatio.types.*;
import org.math.plot.Plot2DPanel;

import javax.swing.*;
import javax.swing.text.html.HTMLDocument;
import java.io.*;

/**
 * Created by ziyihua on 17/07/15.
 */
public class Main {

    public static void main(String[] args) throws IOException {

        Structure.network convnet = new Structure.network();
        for (int i = 0; i < 5; i++) {
            Structure.LAYER layer = new Structure.LAYER();
            convnet.layers.add(i,layer);
        }

        MatFileReader s = new MatFileReader("ss.mat");
        double[][] ss_m = ((MLDouble) s.getMLArray("ss")).getArray();
        float[][][] ss = new float[28][28][10000];
        for (int i = 0; i < 10000; i++) {
            int row=0;
            for (int j = 0; j < 28; j++) {
                for (int k = 0; k < 28; k++) {
                    ss[k][j][i]=(float)ss_m[row][i];
                    row++;
                }
            }
        }


        MatFileReader m = new MatFileReader("cnn.mat");

        MLStructure cnn = (MLStructure) m.getMLArray("cnn");
        MLCell layers = (MLCell) cnn.getField("layers");

        //type
        for (int i = 0; i < 5; i++) {
            convnet.layers.get(i).type= Character.toString(((MLChar) ((MLStructure) layers.get(i)).getField("type")).getChar(0, 0));
        }

        //activation
        MLStructure layer0 = (MLStructure) layers.get(0);
        //layer0
        MLCell layer0_a = (MLCell)layer0.getField("a");
        double[][] layer0_a_m_d = ((MLDouble) layer0_a.get(0)).getArray();
        float[][][] layer0_a_m = new float[28][28][50];
        for (int i = 0; i < 50; i++) {
            for (int j = 0; j < 28; j++) {
                for (int k = 0; k < 28; k++) {
                    layer0_a_m[k][j][i]=(float)layer0_a_m_d[k][i*28+j];
                }
            }
        }
        convnet.layers.get(0).a.add(0,layer0_a_m);
        //layer1
        MLStructure layer1 = (MLStructure) layers.get(1);
        MLCell layer1_a = (MLCell)layer1.getField("a");
        for (int i = 0; i < 6; i++) {
            double[][] layer1_a_m_d = ((MLDouble) layer1_a.get(i)).getArray();
            float[][][] layer1_a_m = new float[24][24][50];
            for (int j = 0; j < 50; j++) {
                for (int k = 0; k < 24; k++) {
                    for (int l = 0; l < 24; l++) {
                        layer1_a_m[l][k][j]=(float)layer1_a_m_d[l][j*24+k];
                    }
                }
            }
            convnet.layers.get(1).a.add(i,layer1_a_m);
        }
        //layer2
        MLStructure layer2 = (MLStructure) layers.get(2);
        MLCell layer2_a = (MLCell)layer2.getField("a");
        for (int i = 0; i < 6; i++) {
            double[][] layer2_a_m_d = ((MLDouble) layer2_a.get(i)).getArray();
            float[][][] layer2_a_m = new float[12][12][50];
            for (int j = 0; j < 50; j++) {
                for (int k = 0; k < 12; k++) {
                    for (int l = 0; l < 12; l++) {
                        layer2_a_m[l][k][j]=(float)layer2_a_m_d[l][j*12+k];
                    }
                }
            }
            convnet.layers.get(2).a.add(i,layer2_a_m);
        }
        //layer3
        MLStructure layer3 = (MLStructure) layers.get(3);
        MLCell layer3_a = (MLCell)layer3.getField("a");
        for (int i = 0; i < 12; i++) {
            double[][] layer3_a_m_d = ((MLDouble) layer3_a.get(i)).getArray();
            float[][][] layer3_a_m = new float[8][8][50];
            for (int j = 0; j < 50; j++) {
                for (int k = 0; k < 8; k++) {
                    for (int l = 0; l < 8; l++) {
                        layer3_a_m[l][k][j]=(float)layer3_a_m_d[l][j*8+k];
                    }
                }
            }
            convnet.layers.get(3).a.add(i,layer3_a_m);
        }
        //layer4
        MLStructure layer4 = (MLStructure) layers.get(4);
        MLCell layer4_a = (MLCell)layer3.getField("a");
        for (int i = 0; i < 12; i++) {
            double[][] layer4_a_m_d = ((MLDouble) layer4_a.get(i)).getArray();
            float[][][] layer4_a_m = new float[4][4][50];
            for (int j = 0; j < 50; j++) {
                for (int k = 0; k < 4; k++) {
                    for (int l = 0; l < 4; l++) {
                        layer4_a_m[l][k][j]=(float)layer4_a_m_d[l][j*4+k];
                    }
                }
            }
            convnet.layers.get(4).a.add(i,layer4_a_m);
        }

        //kernel
        //layer1
        MLCell layer1_k = (MLCell) layer1.getField("k");
        MLCell layer1_k_k = (MLCell) layer1_k.get(0);
        for (int i = 0; i < 6; i++) {
            double[][] layer1_k_m_d = ((MLDouble)layer1_k_k.get(i)).getArray();
            float[][] layer1_k_m = new float[5][5];
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < 5; k++) {
                    layer1_k_m[k][j]=(float)layer1_k_m_d[k][j];
                }
            }
            convnet.layers.get(1).k.add(i,layer1_k_m);
        }
        convnet.layers.get(1).kernelsize=5;
        convnet.layers.get(1).outmaps=6;
        //layer3
        MLCell layer3_k = (MLCell) layer3.getField("k");
        for (int i = 0; i < 6; i++) {
            MLCell layer3_k_k = (MLCell) layer3_k.get(i);
            for (int j = 0; j < 12; j++) {
                double[][] layer3_k_m_d = ((MLDouble)layer3_k_k.get(j)).getArray();
                float[][] layer3_k_m = new float[5][5];
                for (int k = 0; k < 5; k++) {
                    for (int l = 0; l < 5; l++) {
                        layer3_k_m[l][k]=(float)layer3_k_m_d[l][k];
                    }
                }
                convnet.layers.get(3).k.add(i*12+j,layer3_k_m);
            }
        }
        convnet.layers.get(3).kernelsize=5;
        convnet.layers.get(3).outmaps=12;

        //scale
        convnet.layers.get(2).scale=2;
        convnet.layers.get(4).scale=2;

        //ffW
        MLArray ffW = cnn.getField("ffW");
        double[][] ffW_m = ((MLDouble) ffW).getArray();
        convnet.ffW=new float[10][192];
        for (int i = 0; i < ffW_m.length; i++) {
            for (int j = 0; j < ffW_m[0].length; j++) {
                convnet.ffW[i][j]=(float)ffW_m[i][j];
            }
        }

        //fv
        MLArray fv = cnn.getField("fv");
        double[][] fv_m = ((MLDouble) fv).getArray();
        convnet.fv=new float[192][50];
        for (int i = 0; i < fv_m.length; i++) {
            for (int j = 0; j < fv_m[0].length; j++) {
                convnet.fv[i][j]=(float)fv_m[i][j];
            }
        }





/*        MatFileReader m1 = new MatFileReader("k.mat");
        float[][] k_m_d = ((MLfloat) m1.getMLArray("k")).getArray();
        float[][] k_m = new float[k_m_d.length][k_m_d[0].length];
        for (int i = 0; i < k_m.length; i++) {
            for (int j = 0; j < k_m[0].length; j++) {
                k_m[i][j]=(float)k_m_d[i][j];
            }
        }


        MatFileReader m2 = new MatFileReader("ffW.mat");
        float[][] f_m_d = ((MLfloat) m2.getMLArray("ffW")).getArray();
        float[][] f_m = new float[f_m_d.length][f_m_d[0].length];
        for (int i = 0; i < f_m.length; i++) {
            for (int j = 0; j < f_m[0].length; j++) {
                f_m[i][j]=(float)f_m_d[i][j];
            }
        }*/

        /*MatFileReader m3 = new MatFileReader("train_x.mat");
        double[][] train_x = ((MLDouble) m3.getMLArray("train_x")).getArray();
        float[][][] image_t=new float[28][28][60000] ;
        for (int i = 0; i < 60000; i++) {
            int row=0;
            for (int j = 0; j < 28; j++) {
                for (int k = 0; k < 28; k++) {
                    image_t[k][j][i]=(float)train_x[row][i];
                    row++;
                }
            }
        }

        MatFileReader m4 = new MatFileReader("train_y.mat");
        double[][] train_y = ((MLDouble) m4.getMLArray("train_y")).getArray();
        int[][] label_t_new = new int[10][60000];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 60000; j++) {
                label_t_new[i][j]=(int)train_y[i][j];
            }
        }

        MatFileReader m5 = new MatFileReader("kk.mat");
        double[][] kk_m = ((MLDouble) m5.getMLArray("kk")).getArray();
        int[] kk = new int[60000];
        for (int i = 0; i < 60000; i++) {
            kk[i]=(int)kk_m[0][i];
        }*/

        MatFileReader m6 = new MatFileReader("test_x.mat");
        double[][] test_x_m = ((MLDouble) m6.getMLArray("test_x")).getArray();
        float[][][] test_x=new float[28][28][10000] ;
        for (int i = 0; i < 10000; i++) {
            int row=0;
            for (int j = 0; j < 28; j++) {
                for (int k = 0; k < 28; k++) {
                    test_x[k][j][i]=(float)test_x_m[row][i];
                    row++;
                }
            }
        }

        MatFileReader m7 = new MatFileReader("test_y.mat");
        double[][] test_y_m = ((MLDouble) m7.getMLArray("test_y")).getArray();
        int[][] test_y_new = new int[10][10000];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10000; j++) {
                test_y_new[i][j]=(int)test_y_m[i][j];
            }
        }
        int[] test_y = new int[10000];
        for (int i = 0; i < 10000; i++) {
            int j_max=0;
            for (int j = 0; j < 10; j++) {
                if (test_y_new[j][i]==1){
                    j_max=j;
                }
            }
            test_y[i]=j_max;
        }

        //Layers of CNN
        //row 0 is type
        /*String[][] architecture = new String[3][5];
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

        convnet = CNNtrain.CNNTrain(convnet,alpha,numepochs,batchsize,dropout,learn_bias,label_t_new,image_t);


        /*float[] loss = new float[convnet.rL.length];
        for (int i = 0; i < convnet.rL.length; i++) {
            loss[i]=convnet.rL[i];
        }
        int[] indx = new int[loss.length];
        for (int i = 0; i < loss.length; i++) {
            indx[i] = i;
        }*/

        /*int[] test_y;
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
        }*/

        /*System.out.println("1st layer");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < 5; k++) {
                    System.out.println(convnet.layers.get(1).k.get(i)[j][k]);
                }
            }
            System.out.println("lalala");
        }

        System.out.println("ffW");
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 192; j++) {
                System.out.println(convnet.ffW[i][j]);
            }
        }
        System.out.println("3rd layer");
        for (int i = 60; i < 65; i++) {
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < 5; k++) {
                    System.out.println(convnet.layers.get(3).k.get(i)[j][k]);
                }
            }
            System.out.println("lalala");
        }*/


       /* double acc = CNNtest.CNNtest(convnet,test_x,test_y);
        System.out.println(acc);*/


/*        try
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
*/

        /*Plot2DPanel plot = new Plot2DPanel();
        plot.addLegend("SOUTH");

        // add a line plot to the PlotPanel
        plot.addLinePlot("Squared Loss", indx, loss);

        // put the PlotPanel in a JFrame like a JPanel
        JFrame frame = new JFrame("a plot panel");
        frame.setSize(600, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);*/

        /*Structure.network convnet1 = null;
        try {
            FileInputStream fileIn = new FileInputStream("cnn-1.ser");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            convnet1 = (Structure.network) in.readObject();
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
        System.out.println("Deserialized network...");

        Structure.network convnet2 = null;
        try {
            FileInputStream fileIn = new FileInputStream("cnn-2.ser");
            ObjectInputStream in = new ObjectInputStream(fileIn);
            convnet2 = (Structure.network) in.readObject();
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

        /*for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 50; j++) {
                System.out.println(convnet1.o[i][j]-convnet2.o[i][j]);

            }
            
        }*/
        /*for (int i = 0; i < 72; i++) {
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < 5; k++) {
                    System.out.println(convnet1.layers.get(3).k.get(i)[k][j]-convnet2.layers.get(3).k.get(i)[k][j]);
                }
            }
            System.out.println("hahahaha"+i);
        }*/

        convnet = Convlifsim.Convlifsim(convnet,test_x,test_y_new,test_y,0.000f,1.000f,0.001f,0.040f,0.001f,400);

        /*float[] loss_e = new float[e.rL.length];
        for (int i = 0; i < e.rL.length; i++) {
            loss_e[i]=e.rL[i];
        }
        float[] indx_e = new float[loss_e.length];
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
