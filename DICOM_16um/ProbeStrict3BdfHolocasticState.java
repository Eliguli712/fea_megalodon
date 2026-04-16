import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.Arrays;

public class ProbeStrict3BdfHolocasticState {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";

  private static String safe(PropFeature f, String key) {
    try {
      String v = f.getString(key);
      return v == null ? "" : v;
    } catch (Exception ignored) {
      return "";
    }
  }

  private static void dumpMeshFeatureList(String owner, String meshTag, MeshFeatureList fl) {
    try {
      String[] tags = fl.tags();
      System.out.println(owner + "_MESH_FEATURES|" + meshTag + "|" + Arrays.toString(tags));
      for (String t : tags) {
        try {
          PropFeature f = (PropFeature) fl.get(t);
          String type = "";
          try {
            type = f.getType();
          } catch (Exception ignored) {
          }
          String source = safe(f, "source");
          String seq = safe(f, "sequence");
          String fn = safe(f, "filename");
          String createdom = safe(f, "createdom");
          String facepartition = safe(f, "facepartition");
          if (!type.isEmpty() || !source.isEmpty() || !seq.isEmpty() || !fn.isEmpty() || !createdom.isEmpty() || !facepartition.isEmpty()) {
            System.out.println(owner + "_MESH_F|mesh=" + meshTag + "|tag=" + t + "|type=" + type + "|source=" + source + "|sequence=" + seq + "|filename=" + fn + "|createdom=" + createdom + "|facepartition=" + facepartition);
          }
        } catch (Exception ex) {
          System.out.println(owner + "_MESH_FERR|mesh=" + meshTag + "|tag=" + t + "|err=" + ex.getMessage());
        }
      }
    } catch (Exception ex) {
      System.out.println(owner + "_MESH_ERR|" + meshTag + "|" + ex.getMessage());
    }
  }

  public static void main(String[] args) throws Exception {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    System.out.println("MODEL|" + MPH);
    try { System.out.println("COMPONENT_TAGS|" + Arrays.toString(m.component().tags())); } catch (Exception ignored) {}
    try { System.out.println("GLOBAL_MESH_TAGS|" + Arrays.toString(m.mesh().tags())); } catch (Exception ignored) {}

    try {
      for (String mt : m.mesh().tags()) {
        dumpMeshFeatureList("GLOBAL", mt, m.mesh(mt).feature());
      }
    } catch (Exception ignored) {}

    try {
      System.out.println("COMP1_MESH_TAGS|" + Arrays.toString(m.component("comp1").mesh().tags()));
      for (String mt : m.component("comp1").mesh().tags()) {
        dumpMeshFeatureList("COMP1", mt, m.component("comp1").mesh(mt).feature());
      }
    } catch (Exception ignored) {}

    try {
      System.out.println("STUDIES|" + Arrays.toString(m.study().tags()));
      for (String st : m.study().tags()) {
        try {
          String[] map = m.study(st).feature("stat").getStringArray("mesh");
          System.out.println("STUDY_MESH|" + st + "|" + Arrays.toString(map));
        } catch (Exception ex) {
          System.out.println("STUDY_MESH|" + st + "|ERR|" + ex.getMessage());
        }
      }
    } catch (Exception ignored) {}

    try {
      for (String ds : m.result().dataset().tags()) {
        try {
          System.out.println("DSET|" + ds + "|geom=" + m.result().dataset(ds).getString("geom") + "|sol=" + m.result().dataset(ds).getString("solution"));
        } catch (Exception ex) {
          System.out.println("DSET|" + ds + "|ERR|" + ex.getMessage());
        }
      }
    } catch (Exception ignored) {}
  }
}
