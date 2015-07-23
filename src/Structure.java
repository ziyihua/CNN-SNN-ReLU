import org.math.plot.utils.Array;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by ziyihua on 17/07/15.
 */
public class Structure {

    public static class network implements Serializable {
        public network() {
            layers = new ArrayList<LAYER>();
        }

        List<LAYER> layers;
        float[] ffb;
        float[][] ffW;
        float[] rL;
        float acc;
        float[] acc_snn;
        float[][] fv;
        float[][] o;
        float[][] e;
        float L;
        float[][] od;
        float[][] fvd;
        float[][] dffW;
        float[] dffb;
        float first_layer_dropout;
        long[][] time;
        float[][] sum_fv;
        float[][] o_mem;
        float[][] o_refrac_end;
        int[][] o_sum_spikes;
        int[][] o_spikes;
        float[] factor_log;
    }

    public static class LAYER implements Serializable{
        public LAYER() {
            k = new ArrayList<>();
            a = new ArrayList<>();
            d = new ArrayList<>();
            dk = new ArrayList<>();
            m = new ArrayList<>();
            r = new ArrayList<>();
            s = new ArrayList<>();
            sp = new ArrayList<>();

        }

        String type;
        int outmaps;
        int kernelsize;
        int scale;
        List<float[][]> k;
        float[] b;
        List<float[][][]> a;
        List<float[][][]> d;
        List<float[][]> dk;
        List<float[][][]> m;
        List<float[][][]> r;
        List<int[][][]> s;
        List<int[][][]> sp;
        float[] db;
        int[] used_maps;
    }




    public static float[][][] flip3D(float[][][] input){
        float[][][] output = new float[input.length][input[0].length][input[0][0].length];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                for (int k = 0; k < input[0][0].length; k++) {
                    output[i][j][k]=input[input.length-1-i][input[0].length-1-j][input[0][0].length-1-k];
                }
            }
        }
        return output;
    }

}
