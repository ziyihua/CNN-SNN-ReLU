import java.util.Random;

/**
 * Created by ziyihua on 17/07/15.
 */
public class SetUp extends Structure{

    public SetUp(){
    }

    public static network SetUp(String[][] architecture) {
        network convnet = new network();
        int inputmap = 1;
        double mapsize = 28;
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
                int s = Integer.parseInt(architecture[1][i]);
                layer.scale = s;
            }
            if ("c".equals(architecture[0][i])) {
                convnet.layers.add(i, layer);
                convnet.layers.get(i).type = "c";
                int outputmaps = Integer.parseInt(architecture[2][i]);
                layer.outmaps = outputmaps;
                int kernel = Integer.parseInt(architecture[1][i]);
                layer.kernelsize = kernel;
            }
        }
        //initialize random weights and biases
        for (int i = 1; i < architecture[0].length; i++) {
            if ("s".equals(architecture[0][i])) {

                LAYER s_layer = convnet.layers.get(i);

                mapsize = mapsize / (double) s_layer.scale;
                if (Math.floor(mapsize) != mapsize) {
                    System.out.println("Layer" + i + " size must be integer; Actual size is " + mapsize);
                    throw new NumberFormatException();
                }

                double[] b = new double[inputmap];
                for (int j = 0; j < inputmap; j++) {
                    b[j] = 0;
                }
                convnet.layers.get(i).b = b;
            }
            if ("c".equals(architecture[0][i])) {

                LAYER c_layer = convnet.layers.get(i);

                mapsize = mapsize - c_layer.kernelsize + 1;

                //number of features
                double fan_out = c_layer.outmaps * c_layer.kernelsize * c_layer.kernelsize;
                //generate random weights
                int index_w = 0;
                weights w = new weights();
                c_layer.k.add(0, w);
                for (int j = 0; j < c_layer.outmaps; j++) {
                    double fan_in = inputmap * c_layer.kernelsize * c_layer.kernelsize;
                    for (int l = 0; l < inputmap; l++) {
                        double[][] m = new double[c_layer.kernelsize][c_layer.kernelsize];
                        for (int y = 0; y < c_layer.kernelsize; y++) {
                            for (int z = 0; z < c_layer.kernelsize; z++) {
                                Random r = new Random();
                                double randomValue = r.nextDouble();
                                m[y][z] = (randomValue - 0.5) * 2 * Math.sqrt(6.0 / (fan_in + fan_out));
                            }
                        }
                        c_layer.k.get(0).k_list.add(index_w, m);
                        index_w++;
                    }
                }

                double[] b = new double[c_layer.outmaps];
                for (int n = 0; n < c_layer.outmaps; n++) {
                    b[n] = 0;
                }
                convnet.layers.get(i).b = b;

                inputmap = convnet.layers.get(i).outmaps;
            }
        }
        //number of output neuros at the last layer (just before the output layer)
        double fvnum = mapsize*mapsize*inputmap;

        //number of labels
        int onum = 10;

        //biases of the output neurons
        double[] ffb = new double[onum];
        for (int i = 0; i < onum; i++) {
            ffb[i]=0;
        }
        convnet.ffb=ffb;


        //weights between the last layer and the output neurons
        double[][] ffw = new double[onum][(int) fvnum];
        for (int i = 0; i < onum; i++) {
            for (int j = 0; j < fvnum; j++) {
                Random r = new Random();
                double randomValue = r.nextDouble();
                ffw[i][j]=(randomValue - 0.5) * 2 * Math.sqrt(6.0 / (onum + fvnum));
                //ffw[i][j]=randomValue* Math.sqrt(6.0 / (onum + fvnum));
            }
        }
        convnet.ffW=ffw;

        return convnet;
    }
}
