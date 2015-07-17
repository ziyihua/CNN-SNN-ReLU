import java.util.Random;

/**
 * Created by ziyihua on 17/07/15.
 */
public class CNNtrain extends Structure {

    public CNNtrain(){
    }

    public static network CNNTrain(String[][] architecture, network convnet, double alpha, int numepochs, int batchsize, double dropout, int learn_bias, int[][] label, double[][][]image){
        int m=label[0].length;
        int numbatches = m/batchsize;
        if (m % batchsize != 0){
            System.out.println("Number of batches is not integer.");
            throw new NumberFormatException();
        }

        //array storing squared loss
        double[] rl = new double[numepochs*numbatches];
        int loss_index=0;

        for (int i = 0; i < numepochs; i++) {

            System.out.println("Epoch "+i+" / "+numepochs);

            long start = System.currentTimeMillis();

            int[] kk = Permutation.RandomPermutation(m);

            for (int j = 0; j < numbatches; j++) {

                double[][][] batch_x = new double[image.length][image[0].length][batchsize];
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
                            Random r = new Random();
                            double randomValue = r.nextDouble();
                            if (randomValue>dropout){
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
                    rl[loss_index]=convnet.L;
                    loss_index++;
                }else {
                    rl[loss_index]=rl[loss_index-1]*0.99+convnet.L*0.01;
                    loss_index++;
                }
            }


            long end = System.currentTimeMillis();
            long elapsed = end - start;
            long minutes = elapsed / (1000 * 60);
            long seconds = (elapsed / 1000) - (minutes * 60);
            System.out.println("Epoch"+" "+i+" finished:"+" " + minutes + " m " + seconds + " s ");


        }

        convnet.rL=rl;

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
