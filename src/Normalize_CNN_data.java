/**
 * Created by ziyihua on 20/07/15.
 */
public class Normalize_CNN_data extends Structure {

    public Normalize_CNN_data(){
    }

    public static network Normalize_CNN_data(String[][] architecture, network convnet, float[][][] train_x){

        convnet = CNNff.CNNff(architecture,convnet,train_x);

        int previous_factor = 1;

        int inputmaps = 1;

        for (int i = 1; i < convnet.layers.size(); i++) {
            if ("c".equals(convnet.layers.get(i).type)){
                float max_weight = 0.0f;
                float max_activation = 0.0f;
                for (int j = 0; j < inputmaps; j++) {
                    for (int k = 0; k < convnet.layers.get(i).outmaps; k++) {
                        float max_k=0.0f;
                        for (int l = 0; l < convnet.layers.get(i).kernelsize; l++) {
                            for (int m = 0; m < convnet.layers.get(i).kernelsize; m++) {
                                if (convnet.layers.get(i).k.get(j*convnet.layers.get(i).outmaps+k)[l][m]>max_k){
                                    max_k=convnet.layers.get(i).k.get(j*convnet.layers.get(i).outmaps+k)[l][m];
                                }
                            }
                        }
                        if (max_k>max_weight){
                            max_weight=max_k;
                        }
                    }
                }
            }
        }



        return convnet;
    }
}
