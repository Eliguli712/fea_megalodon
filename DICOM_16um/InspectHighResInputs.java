import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;

public class InspectHighResInputs {
  private static final String MODEL_PATH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics_high_resolution.mph";

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MODEL_PATH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load: " + MODEL_PATH, e);
    }

    for (String partTag : model.mesh().tags()) {
      try {
        String fn = model.mesh(partTag).feature("imp1").getString("filename");
        String src = model.mesh(partTag).feature("imp1").getString("source");
        System.out.println("IMP|" + partTag + "|source=" + src + "|file=" + fn);
      } catch (Exception e) {
        System.out.println("IMP|" + partTag + "|error=" + e.getMessage());
      }
    }
  }
}
