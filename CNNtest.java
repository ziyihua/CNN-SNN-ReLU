/**
 * Created by ziyihua on 17/07/15.
 */
public class CNNtest extends Structure {

    public CNNtest(){
    }

    public static double CNNtest(String[][] architecture, network convnet, double[][][]test_x, int[]test_y){

        CNNff.CNNff(architecture,convnet,test_x);
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

        return (double) correct/test_y.length;
    }

}
