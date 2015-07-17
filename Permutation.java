/**
 * Created by ziyihua on 17/07/15.
 */
public class Permutation {

    public static int[] RandomPermutation(int N){
        int[] perm_array=new int [N];

        for (int i = 0; i < N; i++)
            perm_array[i] = i;

        for (int i = 0; i < N; i++) {
            int r = (int) (Math.random() * (i+1));     // int between 0 and i
            int swap = perm_array[r];
            perm_array[r] = perm_array[i];
            perm_array[i] = swap;
        }

        return perm_array;
    }

}
