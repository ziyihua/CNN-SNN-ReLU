import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Created by ziyihua on 17/07/15.
 */
public class ImportFile {

    public static int[] getLabel(String filename) throws IOException {
        DataInputStream labels = new DataInputStream(new FileInputStream(filename));

        int magicNumber = labels.readInt();
        if (magicNumber != 2049) {
            System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
            System.exit(0);
        }

        int numLabels = labels.readInt();

        int numLabelsRead = 0;

        int[] label_t = new int[numLabels];

        while (labels.available() > 0 && numLabelsRead < numLabels) {
            byte label = labels.readByte();
            label_t[numLabelsRead] = label;
            numLabelsRead++;
        }

        return label_t;
    }

    public static float[][][] getImage(String filename) throws IOException {
        DataInputStream images = new DataInputStream(new FileInputStream(filename));

        int magicNumber = images.readInt();
        if (magicNumber != 2051) {
            System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
            System.exit(0);
        }

        int numImages = images.readInt();
        int numRows = images.readInt();
        int numCols = images.readInt();

        int numImagesRead = 0;
        float[][][] image_t = new float[28][28][numImages];
        while (images.available() > 0 && numImagesRead < numImages) {
            for (int colIdx = 0; colIdx < numCols; colIdx++) {
                for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    image_t[colIdx][rowIdx][numImagesRead] = images.readUnsignedByte();
                    image_t[colIdx][rowIdx][numImagesRead] = image_t[colIdx][rowIdx][numImagesRead]/255;
                }
            }
            numImagesRead++;
        }
        return image_t;
    }

}
