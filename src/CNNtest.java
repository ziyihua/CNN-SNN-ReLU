/**
 * Created by ziyihua on 17/07/15.
 */
public class CNNtest extends Structure {

    public CNNtest(){
    }

    public static double CNNtest(network convnet, float[][][]test_x, int[]test_y){

        convnet = CNNff.CNNff(convnet,test_x);
        int[] output = new int[convnet.o[0].length];
        for (int i = 0; i < convnet.o[0].length; i++) {
            double max = 0;
            int j_max = 0;
            for (int j = 0; j < convnet.o.length; j++) {
                if (convnet.o[j][i]>max){
                    max=convnet.o[j][i];
                    j_max=j;
                }
            }
            output[i]=j_max;
        }

        int correct = 0;
        for (int i = 0; i < test_y.length ; i++) {
            if (output[i]==test_y[i])
                correct++;
        }

        System.out.println(correct);

        return (double)correct/(double)test_y.length;

    }

}
