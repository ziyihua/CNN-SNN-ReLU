/**
 * Created by ziyihua on 23/07/15.
 */
public class Normalize_CNN_model extends Structure {

    public Normalize_CNN_model(){
    }

    public static network Normalize_CNN_model (network convnet, int strong_norm){

        for (int i = 1; i < convnet.layers.size(); i++) {

            if ("c".equals(convnet.layers.get(i).type)){

                float[] weight_sum = new float[convnet.layers.get(i).a.size()];
                for (int j = 0; j < weight_sum.length; j++) {
                    weight_sum[j]=0;
                }

                for (int j = 0; j < convnet.layers.get(i).a.size(); j++) {
                    for (int k = 0; k < convnet.layers.get(i-1).a.size(); k++) {
                        for (int l = 0; l < convnet.layers.get(i).kernelsize; l++) {
                            for (int m = 0; m < convnet.layers.get(i).kernelsize; m++) {
                                if (convnet.layers.get(i).k.get(k*convnet.layers.get(i).outmaps+j)[l][m]>0){
                                    weight_sum[j]=weight_sum[j]+convnet.layers.get(i).k.get(k*convnet.layers.get(i).outmaps+j)[l][m];
                                }
                            }
                        }
                    }
                }

                float max_weight_sum=0;
                for (int j = 0; j < weight_sum.length; j++) {
                    if (weight_sum[j]>max_weight_sum){
                        max_weight_sum=weight_sum[j];
                    }
                }

                if (strong_norm==1){
                    for (int j = 0; j < convnet.layers.get(i).k.size(); j++) {
                        float[][] new_k = new float[convnet.layers.get(i).kernelsize][convnet.layers.get(i).kernelsize];
                        for (int k = 0; k < new_k.length; k++) {
                            for (int l = 0; l < new_k[0].length; l++) {
                                new_k[k][l]=convnet.layers.get(i).k.get(j)[k][l]/max_weight_sum;
                            }
                        }
                        convnet.layers.get(i).k.set(j,new_k);
                    }
                }else {
                    if(max_weight_sum>1){
                        for (int j = 0; j < convnet.layers.get(i).k.size(); j++) {
                            float[][] new_k = new float[convnet.layers.get(i).kernelsize][convnet.layers.get(i).kernelsize];
                            for (int k = 0; k < new_k.length; k++) {
                                for (int l = 0; l < new_k[0].length; l++) {
                                    new_k[k][l]=convnet.layers.get(i).k.get(j)[k][l]/max_weight_sum;
                                }
                            }
                            convnet.layers.get(i).k.set(j,new_k);
                        }
                    }
                }
            }
        }
        return convnet;
    }

}
