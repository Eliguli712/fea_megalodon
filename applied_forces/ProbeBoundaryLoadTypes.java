import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeBoundaryLoadTypes {
  public static Model run() {
    String mphPath = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    Model model;
    try {
      model = ModelUtil.load("Model", mphPath);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    try {
      model.component("comp1").physics("solid").feature("bndl_pr");
      String[] ops = new String[] {
        "Pressure", "BoundaryPressure", "PressureLoad", "PressureBoundary", "BoundaryPressureLoad", "pr"
      };
      for (String op : ops) {
        String t = "tmp_" + op.replaceAll("[^A-Za-z0-9]", "").toLowerCase();
        try {
          model.component("comp1").physics("solid").create(t, op, 2);
          System.out.println("create op '" + op + "' succeeded as tag " + t);
        } catch (Exception e) {
          System.out.println("create op '" + op + "' failed: " + e.getMessage());
        }
      }

      try {
        String[] allowed = model.component("comp1").physics("solid").feature("bndl_pr").getAllowedPropertyValues("forceType");
        if (allowed != null) {
          System.out.println("allowed forceType values:");
          for (String a : allowed) System.out.println("  " + a);
        } else {
          System.out.println("allowed forceType values: null");
        }
      } catch (Exception e) {
        System.out.println("could not query allowed forceType values: " + e.getMessage());
      }

      try {
        String[] allowed = model.component("comp1").physics("solid").feature("bndl_pr").getAllowedPropertyValues("LoadType");
        if (allowed != null) {
          System.out.println("allowed LoadType values:");
          for (String a : allowed) System.out.println("  " + a);
        } else {
          System.out.println("allowed LoadType values: null");
        }
      } catch (Exception e) {
        System.out.println("could not query allowed LoadType values: " + e.getMessage());
      }

      String[] vals = new String[] {"Pressure", "pressure", "ForceArea", "ForcePerArea", "NormalPressure", "LoadPressure"};
      for (String v : vals) {
        try {
          model.component("comp1").physics("solid").feature("bndl_pr").set("forceType", v);
          String out = model.component("comp1").physics("solid").feature("bndl_pr").getString("forceType");
          System.out.println("forceType set '" + v + "' -> '" + out + "'");
        } catch (Exception e) {
          System.out.println("forceType rejected '" + v + "' : " + e.getMessage());
        }
      }

      String[] lvals = new String[] {"ForceArea", "Pressure", "ForceVolume", "ForceLength", "ForcePoint"};
      for (String v : lvals) {
        try {
          model.component("comp1").physics("solid").feature("bndl_pr").set("LoadType", v);
          String out = model.component("comp1").physics("solid").feature("bndl_pr").getString("LoadType");
          System.out.println("LoadType set '" + v + "' -> '" + out + "'");
        } catch (Exception e) {
          System.out.println("LoadType rejected '" + v + "' : " + e.getMessage());
        }
      }

      String[] pvals = new String[] {"thrust_load", "-thrust_load", "5e4[Pa]"};
      for (String pv : pvals) {
        try {
          model.component("comp1").physics("solid").feature("bndl_pr").set("pressure", pv);
          String out = model.component("comp1").physics("solid").feature("bndl_pr").getString("pressure");
          System.out.println("pressure set '" + pv + "' -> '" + out + "'");
        } catch (Exception e) {
          System.out.println("pressure rejected '" + pv + "' : " + e.getMessage());
        }
      }

      String[] keys = new String[] {"LoadType", "loadType", "Pressure", "p0", "FperArea", "force", "tractionType"};
      for (String k : keys) {
        try {
          String out = model.component("comp1").physics("solid").feature("bndl_pr").getString(k);
          System.out.println("getString(" + k + ") -> " + out);
        } catch (Exception e) {
          System.out.println("no key " + k + " : " + e.getMessage());
        }
      }
    } catch (Exception e) {
      System.out.println("bndl_pr not present: " + e.getMessage());
    }

    return model;
  }

  public static void main(String[] args) {
    run();
  }
}
