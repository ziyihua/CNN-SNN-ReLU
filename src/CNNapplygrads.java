import java.util.ArrayList;

/**
 * Created by ziyihua on 17/07/15.
 */
public class CNNapplygrads extends Structure {

    public CNNapplygrads(){
    }

    public static network CNNapplygrads(network convnet, float alpha, int learn_bias){

        int n = convnet.layers.size();

        for (int i = 1; i < n; i++) {
            if("c".equals((convnet.layers.get(i).type))){

                LAYER layer_current = convnet.layers.get(i);
                LAYER layer_previous = convnet.layers.get(i - 1);

                int num_a_current = layer_current.a.size();
                int num_a_previous = layer_previous.a.size();

                //update weights
                for (int k = 0; k < num_a_previous; k++) {
                    for (int j = 0; j < num_a_current; j++) {
                        float[][] kernel_new = new float[layer_current.kernelsize][layer_current.kernelsize];
                        for (int l = 0; l < layer_current.kernelsize; l++) {
                            for (int m = 0; m < layer_current.kernelsize; m++) {
                                kernel_new[l][m] = layer_current.k.get(j+k*layer_current.outmaps)[l][m]-alpha*layer_current.dk.get(j+k*layer_current.outmaps)[l][m];
                            }
                        }
                        layer_current.k.set(j+k*layer_current.outmaps,kernel_new);
                    }
                }

                if(learn_bias != 0) {
                    //update biases
                    for (int j = 0; j < num_a_current; j++) {
                        layer_current.b[j] = layer_current.b[j] - alpha * layer_current.db[j];
                    }
                }
            }
        }

        //update ffW
        for (int i = 0; i < convnet.ffW.length; i++) {
            for (int j = 0; j < convnet.ffW[0].length; j++) {
                convnet.ffW[i][j]=convnet.ffW[i][j]-alpha*convnet.dffW[i][j];
            }
        }

        //update ffb
        for (int i = 0; i < convnet.ffb.length; i++) {
            convnet.ffb[i]=convnet.ffb[i]-alpha*convnet.dffb[i];
        }

        return convnet;
    }
}
