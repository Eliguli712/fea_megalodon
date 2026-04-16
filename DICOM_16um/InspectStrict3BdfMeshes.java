import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.Arrays;

public class InspectStrict3BdfMeshes {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution.mph";

  private static String s(PropFeature f, String key) {
    try {
      String v = f.getString(key);
      return v == null ? "" : v;
    } catch (Exception ignored) {
      return "";
    }
  }

  private static void dumpMeshFeatureList(String label, MeshFeatureList fl) {
    try {
      String[] tags = fl.tags();
      System.out.println(label + "_FEATURES|" + Arrays.toString(tags));
      for (String t : tags) {
        try {
          PropFeature f = (PropFeature) fl.get(t);
          String type = "";
          try {
            type = f.getType();
          } catch (Exception ignored) {
          }
          String source = s(f, "source");
          String seq = s(f, "sequence");
          String filename = s(f, "filename");
          String createdom = s(f, "createdom");
          if (!source.isEmpty() || !seq.isEmpty() || !filename.isEmpty() || !createdom.isEmpty() || "Import".equals(type)) {
            System.out.println(
                label
                    + "_F|tag="
                    + t
                    + "|type="
                    + type
                    + "|source="
                    + source
                    + "|sequence="
                    + seq
                    + "|filename="
                    + filename
                    + "|createdom="
                    + createdom);
          }
        } catch (Exception ex) {
          System.out.println(label + "_FERR|tag=" + t + "|err=" + ex.getMessage());
        }
      }
    } catch (Exception ex) {
      System.out.println(label + "_ERR|" + ex.getMessage());
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
    for (String tag : model.mesh().tags()) {
      dumpMeshFeatureList("GLOBAL_" + tag, model.mesh(tag).feature());
    }

    System.out.println("COMP1_MESH_TAGS|" + Arrays.toString(model.component("comp1").mesh().tags()));
    for (String tag : model.component("comp1").mesh().tags()) {
      dumpMeshFeatureList("COMP1_" + tag, model.component("comp1").mesh(tag).feature());
    }
  }
}
