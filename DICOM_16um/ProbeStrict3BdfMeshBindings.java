import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.Arrays;

public class ProbeStrict3BdfMeshBindings {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";

  private static String safeGet(PropFeature pf, String key) {
    try {
      String v = pf.getString(key);
      return v == null ? "" : v;
    } catch (Exception ignored) {
      return "";
    }
  }

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    System.out.println("MODEL|" + MPH);
    System.out.println("GLOBAL_MESH_TAGS|" + Arrays.toString(model.mesh().tags()));
    try {
      System.out.println("MPART1_IMP1_FILE|" + safeGet(model.mesh("mpart1").feature("imp1"), "filename"));
    } catch (Exception e) {
      System.out.println("MPART1_IMP1_FILE|ERR|" + e.getMessage());
    }

    System.out.println("COMP1_MESH_TAGS|" + Arrays.toString(model.component("comp1").mesh().tags()));
    for (String std : model.study().tags()) {
      try {
        String[] map = model.study(std).feature("stat").getStringArray("mesh");
        System.out.println("STUDY_MESH|" + std + "|" + Arrays.toString(map));
      } catch (Exception e) {
        System.out.println("STUDY_MESH|" + std + "|ERR|" + e.getMessage());
      }
    }
  }
}
