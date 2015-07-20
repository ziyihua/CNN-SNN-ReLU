import java.io.IOException;
import java.util.Random;

/**
 * Created by ziyihua on 17/07/15.
 */
public class CNNtrain extends Structure {

    public CNNtrain(){
    }

    public static network CNNTrain(String[][] architecture, network convnet, float alpha, int numepochs, int batchsize, float dropout, int learn_bias, int[][] label, float[][][]image){
        int m=label[0].length;
        int numbatches = m/batchsize;
        if (m % batchsize != 0){
            System.out.println("Number of batches is not integer.");
            throw new NumberFormatException();
        }

        Random r = new Random(0);

        //array storing squared loss
        convnet.rL = new float[numepochs*numbatches];
        int loss_index=0;
        long[][] time = new long[2][numepochs];

        for (int i = 0; i < numepochs; i++) {

            if(i==1 && convnet.rL[loss_index-1]>0.5){
                throw new NumberFormatException();
            }

            System.out.println("Epoch "+i+" / "+numepochs);

            long start = System.currentTimeMillis();

            int[] kk = Permutation.RandomPermutation(m);

            for (int j = 0; j < numbatches; j++) {

                float[][][] batch_x = new float[image.length][image[0].length][batchsize];
                for (int k = 0; k <image.length ; k++) {
                    for (int l = 0; l < image[0].length ; l++) {
                        for (int n = 0; n < batchsize; n++) {
                            batch_x[k][l][n]=image[k][l][kk[n+j*batchsize]];
                        }
                    }
                }

                int[][] batch_y = new int[label.length][batchsize];
                for (int k = 0; k < label.length; k++) {
                    for (int l = 0; l < batchsize ; l++) {
                        batch_y[k][l]=label[k][kk[l+j*batchsize]];
                    }
                }

                //randomly disable input units with probability=dropout
                for (int k = 1; k < convnet.layers.size(); k++) {
                    if ("c".equals(architecture[0][k])){
                        int num_maps = convnet.layers.get(k).outmaps;
                        int[] indx = new int[num_maps];
                        for (int l = 0; l < indx.length; l++) {
                            //Random r = new Random();
                            double randomValue = r.nextDouble();
                            if ((float)randomValue>dropout){
                                indx[l]=1;
                            }else indx[l]=0;
                        }
                        convnet.layers.get(k).used_maps=indx;
                    }
                }


                convnet = CNNff.CNNff(architecture, convnet, batch_x);

                convnet = CNNbp.CNNbp(architecture, convnet, batch_y);

                convnet = CNNapplygrads.CNNapplygrads(architecture,convnet,alpha,learn_bias);

                if (loss_index==0){
                    convnet.rL[loss_index]=convnet.L;
                    loss_index++;
                }else {
                    convnet.rL[loss_index]=(float)(convnet.rL[loss_index-1]*0.99+convnet.L*0.01);
                    loss_index++;
                }
            }


            long end = System.currentTimeMillis();
            long elapsed = end - start;
            long minutes = elapsed / (1000 * 60);
            long seconds = (elapsed / 1000) - (minutes * 60);
            time[0][i]=minutes;
            time[1][i]=seconds;
            System.out.println("Epoch"+" "+i+" finished:"+" " + minutes + " m " + seconds + " s ");

        }

        convnet.time=time;

        for (int k = 1; k < convnet.layers.size(); k++) {
            if ("c".equals(architecture[0][k])){
                int num_maps = convnet.layers.get(k).outmaps;
                int[] indx = new int[num_maps];
                for (int l = 0; l < indx.length; l++) {
                    indx[l]=1;
                }
                convnet.layers.get(k).used_maps=indx;
            }
        }

        return convnet;

    }

}
