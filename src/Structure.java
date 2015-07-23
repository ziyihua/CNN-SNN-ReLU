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
        double[] ffb;
        double[][] ffW;
        double[] rL;
        double acc;
        double[] acc_snn;
        double[][] fv;
        double[][] o;
        double[][] e;
        double L;
        double[][] od;
        double[][] fvd;
        double[][] dffW;
        double[] dffb;
        double first_layer_dropout;
        long[][] time;
        double[][] sum_fv;
        double[][] o_mem;
        double[][] o_refrac_end;
        double[][] o_sum_spikes;
        double[][] o_spikes;
        double[] factor_log;
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
        List<double[][]> k;
        double[] b;
        List<double[][][]> a;
        List<double[][][]> d;
        List<double[][]> dk;
        List<double[][][]> m;
        List<double[][][]> r;
        List<double[][][]> s;
        List<double[][][]> sp;
        double[] db;
        int[] used_maps;
    }




    public static double[][][] flip3D(double[][][] input){
        double[][][] output = new double[input.length][input[0].length][input[0][0].length];
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
