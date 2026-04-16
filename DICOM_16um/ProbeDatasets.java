import com.comsol.model.*;
import com.comsol.model.util.*;

public class ProbeDatasets {
  private static final String MODEL_PATH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics.mph";

  public static void main(String[] args) throws Exception {
    Model model = ModelUtil.load("Model", MODEL_PATH);
    String[] tags = model.result().dataset().tags();
    for (String tag : tags) {
      String label = "";
      String type = "";
      String data = "";
      try {
        label = model.result().dataset(tag).label();
      } catch (Exception ignored) {
      }
      try {
        type = model.result().dataset(tag).getType();
      } catch (Exception ignored) {
      }
      try {
        data = model.result().dataset(tag).getString("data");
      } catch (Exception ignored) {
      }
      System.out.println("DATASET|" + tag + "|" + type + "|" + label + "|data=" + data);
    }
  }
}
