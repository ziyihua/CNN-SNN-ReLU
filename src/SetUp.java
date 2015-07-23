import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;

import java.util.Random;

/**
 * Created by ziyihua on 17/07/15.
 */
public class SetUp extends Structure{

    public SetUp(){
    }

    public static network SetUp (String[][] architecture) {
        Random r = new Random(9);
        network convnet = new network();
        int inputmap = 1;
        float mapsize = 28;
        //add input layer
        LAYER inputlayer = new LAYER();
        convnet.layers.add(0, inputlayer);
        convnet.layers.get(0).type = "i";
        //add convolution and subsampling layers
        for (int i = 1; i < architecture[0].length; i++) {
            LAYER layer = new LAYER();
            if ("s".equals(architecture[0][i])) {
                convnet.layers.add(i, layer);
                convnet.layers.get(i).type = "s";
                layer.scale = Integer.parseInt(architecture[1][i]);
            }
            if ("c".equals(architecture[0][i])) {
                convnet.layers.add(i, layer);
                convnet.layers.get(i).type = "c";
                layer.outmaps = Integer.parseInt(architecture[2][i]);
                layer.kernelsize = Integer.parseInt(architecture[1][i]);
            }
        }
        int k_indx = 0;
        //initialize random weights and biases
        for (int i = 1; i < architecture[0].length; i++) {
            if ("s".equals(architecture[0][i])) {

                LAYER s_layer = convnet.layers.get(i);

                mapsize = mapsize / (float) s_layer.scale;
                if (Math.floor(mapsize) != mapsize) {
                    System.out.println("Layer" + i + " size must be integer; Actual size is " + mapsize);
                    throw new NumberFormatException();
                }

                convnet.layers.get(i).b = new float[inputmap];
                for (int j = 0; j < inputmap; j++) {
                    convnet.layers.get(i).b[j] = 0;
                }

            }
            if ("c".equals(architecture[0][i])) {

                LAYER c_layer = convnet.layers.get(i);

                mapsize = mapsize - c_layer.kernelsize + 1;

                //number of features
                float fan_out = c_layer.outmaps * c_layer.kernelsize * c_layer.kernelsize;
                //generate random weights
                int index_w = 0;
                for (int j = 0; j < c_layer.outmaps; j++) {
                    float fan_in = inputmap * c_layer.kernelsize * c_layer.kernelsize;
                    for (int l = 0; l < inputmap; l++) {
                        float[][] m = new float[c_layer.kernelsize][c_layer.kernelsize];
                        for (int y = 0; y < c_layer.kernelsize; y++) {
                            for (int z = 0; z < c_layer.kernelsize; z++) {
                                float randomValue = r.nextFloat();
                                m[y][z] = ((float)(randomValue - 0.5)) * 2 * ((float)Math.sqrt(6.0 / (fan_in + fan_out)));
                            }
                        }
                        c_layer.k.add(index_w,m);
                        index_w++;
                    }
                }
                /*for (int j = 0; j < inputmap; j++) {
                    for (int k = 0; k < c_layer.outmaps; k++) {
                        float[][] m = new float[c_layer.kernelsize][c_layer.kernelsize];
                        for (int y = 0; y < c_layer.kernelsize; y++) {
                            for (int z = 0; z < c_layer.kernelsize; z++) {
                                m[y][z] = k_m[k_indx][z];
                            }
                            k_indx++;
                        }
                        c_layer.k.add(index_w,m);
                        index_w++;
                    }
                }*/

                convnet.layers.get(i).b = new float[c_layer.outmaps];
                for (int n = 0; n < c_layer.outmaps; n++) {
                    convnet.layers.get(i).b[n] = 0;
                }

                inputmap = convnet.layers.get(i).outmaps;
            }
        }
        //number of output neuros at the last layer (just before the output layer)
        float fvnum = mapsize*mapsize*inputmap;
        //number of labels
        int onum = 10;

        //biases of the output neurons
        convnet.ffb = new float[onum];
        for (int i = 0; i < onum; i++) {
            convnet.ffb[i]=0.0f;
        }



        //weights between the last layer and the output neurons
        convnet.ffW = new float[onum][(int) fvnum];
        for (int i = 0; i < onum; i++) {
            for (int j = 0; j < fvnum; j++) {
                //Random r = new Random();
                float randomValue = r.nextFloat();
                convnet.ffW[i][j]=(float)(randomValue * Math.sqrt(6.0 / (onum + fvnum)));
            }
        }
        /*for (int i = 0; i < onum; i++) {
            for (int j = 0; j < fvnum; j++) {
                convnet.ffW[i][j]=f_m[i][j];
            }
        }*/

        return convnet;
    }
}
