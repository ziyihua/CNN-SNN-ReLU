import java.util.ArrayList;

/**
 * Created by ziyihua on 17/07/15.
 */
public class CNNapplygrads extends Structure {

    public CNNapplygrads(){
    }

    public static network CNNapplygrads(String[][] architecture, network convnet, double alpha, int learn_bias){

        int n = convnet.layers.size();

        for (int i = 1; i < n; i++) {
            if("c".equals((architecture[0][i]))){

                LAYER layer_current = convnet.layers.get(i);
                LAYER layer_previous = convnet.layers.get(i-1);

                int num_a_current = layer_current.a.get(0).a_list.size();
                int num_a_previous = layer_previous.a.get(0).a_list.size();

                ArrayList k_new = new ArrayList();
                double[] b_new = new double[layer_current.b.length];

                //update weights
                for (int k = 0; k < num_a_previous; k++) {
                    for (int j = 0; j < num_a_current; j++) {
                        double[][] kernel_new = new double[layer_current.kernelsize][layer_current.kernelsize];
                        for (int l = 0; l < layer_current.kernelsize; l++) {
                            for (int m = 0; m < layer_current.kernelsize; m++) {
                                kernel_new[l][m] = ((double[][])layer_current.k.get(0).k_list.get(j+k*layer_current.outmaps))[l][m]-alpha* ((double[][])layer_current.dk.get(0).dk_list.get(j+k*layer_current.outmaps))[l][m];
                            }
                        }
                        k_new.add(j+k*layer_current.outmaps,kernel_new);
                    }
                }

                if(learn_bias != 0) {
                    //update biases
                    for (int j = 0; j < num_a_current; j++) {
                        b_new[j] = layer_current.b[j] - alpha * layer_current.db[j];
                    }
                }

                layer_current.b=b_new;
                layer_current.k.get(0).k_list.clear();
                layer_current.k.get(0).k_list.addAll(k_new);
            }
        }

        //update ffW
        double[][] ffW = new double[convnet.ffW.length][convnet.ffW[0].length];
        for (int i = 0; i < convnet.ffW.length; i++) {
            for (int j = 0; j < convnet.ffW[0].length; j++) {
                ffW[i][j]=convnet.ffW[i][j]-alpha*convnet.dffW[i][j];
            }
        }
        convnet.ffW=ffW;

        //update ffb
        double[] ffb = new double[convnet.ffb.length];
        for (int i = 0; i < convnet.ffb.length; i++) {
            ffb[i]=convnet.ffb[i]-alpha*convnet.dffb[i];
        }
        convnet.ffb=ffb;

        return convnet;
    }
}
