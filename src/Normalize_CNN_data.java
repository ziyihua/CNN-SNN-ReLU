import java.util.DoubleSummaryStatistics;

/**
 * Created by ziyihua on 20/07/15.
 */
public class Normalize_CNN_data extends Structure {

    public Normalize_CNN_data(){
    }

    public static network Normalize_CNN_data(network convnet, float[][][] train_x){

        convnet = CNNff.CNNff(convnet,train_x);

        float previous_factor = 1;

        convnet.factor_log = new float[convnet.layers.size()];
        for (int i = 0; i < convnet.factor_log.length; i++) {
           convnet.factor_log[i]= (float)Double.NaN;
        }

        for (int i = 1; i < convnet.layers.size(); i++) {
            if ("c".equals(convnet.layers.get(i).type)){

                float max_weight = 0.0f;
                float max_activation = 0.0f;

                for (int j = 0; j < convnet.layers.get(i).k.size(); j++) {
                    for (int k = 0; k < convnet.layers.get(i).kernelsize; k++) {
                        for (int l = 0; l < convnet.layers.get(i).kernelsize; l++) {
                            if (convnet.layers.get(i).k.get(j)[k][l]>max_weight){
                                max_weight=convnet.layers.get(i).k.get(j)[k][l];
                            }
                        }
                    }
                }

                for (int j = 0; j < convnet.layers.get(i).a.size(); j++) {
                    for (int k = 0; k < convnet.layers.get(i).a.get(0).length; k++) {
                        for (int l = 0; l < convnet.layers.get(i).a.get(0)[0].length; l++) {
                            for (int m = 0; m < convnet.layers.get(i).a.get(0)[0][0].length; m++) {
                                if (convnet.layers.get(i).a.get(j)[k][l][m]>max_activation){
                                    max_activation=convnet.layers.get(i).a.get(j)[k][l][m];
                                }
                            }
                        }
                    }
                }

                float scale_factor;
                if (max_activation>max_weight){
                    scale_factor=max_activation;
                }else {
                    scale_factor=max_weight;
                }

                float current_factor = scale_factor/previous_factor;


                for (int j = 0; j < convnet.layers.get(i).k.size(); j++) {
                    float[][] new_k = new float[convnet.layers.get(i).kernelsize][convnet.layers.get(i).kernelsize];
                    for (int k = 0; k < convnet.layers.get(i).kernelsize; k++) {
                        for (int l = 0; l < convnet.layers.get(i).kernelsize; l++) {
                            new_k[k][l]=convnet.layers.get(i).k.get(j)[k][l]/current_factor;
                        }
                    }
                    convnet.layers.get(i).k.set(j,new_k);
                }
                convnet.factor_log[i]=1/current_factor;
                previous_factor=current_factor;
            }
        }
        return convnet;
    }
}
