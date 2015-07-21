import java.util.Random;

/**
 * Created by ziyihua on 21/07/15.
 */
public class RandomNumberGenerator {
    static long[][] array = new long[2][64];

    public RandomNumberGenerator(int t){
        Random r = new Random(t);
        for (int i = 0; i < 63; i++) {
            array[0][i]=i;
            array[1][i]=(long)(r.nextDouble() * ((Math.pow(2,31) + 1)));
        }
    }

    public static float NextFloat(){
        int max_pos = 0;

        for (int i = 0; i < 62; i++) {
            if(array[0][i]<array[0][i+1]){
                max_pos=i+1;
            }
        }

        long max_indx=array[0][max_pos];

        if(max_pos==62){
            array[0][1]=max_indx+1;
            array[1][1]= (long)Math.pow(2,63)-array[1][1]*array[1][32];
        }else {
            array[0][max_pos+1]=max_indx+1;
            int indx = 0;
            for (int i = 0; i < 62; i++) {
                if (array[0][i]==max_pos-30){
                    indx = i;
                }
            }
            array[1][max_pos+1]=(long)Math.pow(2,62)-array[1][max_pos+1]*array[1][indx];
            System.out.println(array[1][max_pos+1]);
        }
        return (float)(array[1][max_pos+1]/Math.pow(2,62));
    }


}
