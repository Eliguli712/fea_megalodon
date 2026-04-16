import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class RunHolocasticMesh2ComputeOnly {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
  private static final String BDF_CONFORMING =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol_noperson2_fixinv1.bdf";

  private static boolean hasStudy(Model m, String tag) {
    try {
      m.study(tag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static String datasetForStudy(String studyTag) {
    if ("std1".equals(studyTag)) return "dset6";
    if ("std_nh".equals(studyTag)) return "dset1";
    if ("std_og".equals(studyTag)) return "dset2";
    if ("std_mr2".equals(studyTag)) return "dset3";
    if ("std_mr5".equals(studyTag)) return "dset4";
    if ("std_pr".equals(studyTag)) return "dset5";
    return "dset6";
  }

  private static double evalMaxMises(Model m, String dataset, String tag) {
    try {
      try {
        m.result().numerical().remove(tag);
      } catch (Exception ignored) {
      }
      m.result().numerical().create(tag, "MaxVolume");
      m.result().numerical(tag).set("expr", new String[]{"solid.mises"});
      m.result().numerical(tag).set("unit", new String[]{"Pa"});
      m.result().numerical(tag).set("data", dataset);
      m.result().numerical(tag).selection().all();
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) {
        return r[0][0];
      }
    } catch (Exception ignored) {
    }
    return Double.NaN;
  }

  public static void main(String[] args) {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed loading model: " + MPH, e);
    }

    List<String> logs = new ArrayList<String>();
    logs.add("MODEL|" + MPH);

    // Refresh mesh import and force mesh part 2 build.
    try {
      m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "nastran");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("filename", BDF_CONFORMING);
      m.component("comp1").mesh("mesh1").run("impmsh");
      logs.add("MESH1_IMPORT|ok=true");
    } catch (Exception e) {
      logs.add("MESH1_IMPORT|ok=false|err=" + e.getMessage());
    }

    try {
      m.component("comp1").mesh("mesh2").feature("size1").selection().geom("geom1", 3);
      m.component("comp1").mesh("mesh2").feature("size1").selection().all();
    } catch (Exception ignored) {
    }
    try {
      m.component("comp1").mesh("mesh2").feature("ftet1").selection().geom("geom1", 3);
      m.component("comp1").mesh("mesh2").feature("ftet1").selection().all();
    } catch (Exception ignored) {
    }
    try {
      m.component("comp1").mesh("mesh2").run();
      logs.add("MESH2_RUN|ok=true");
    } catch (Exception e) {
      logs.add("MESH2_RUN|ok=false|err=" + e.getMessage());
    }

    try {
      int dom = m.component("comp1").mesh("mesh2").getNumElem("tet");
      logs.add("MESH2_TET_COUNT|" + dom);
    } catch (Exception ignored) {
    }

    String[] studies = new String[]{"std1", "std_nh", "std_og", "std_mr2", "std_mr5", "std_pr"};
    for (String st : studies) {
      if (!hasStudy(m, st)) {
        logs.add("STUDY|" + st + "|status=missing");
        continue;
      }
      try {
        m.study(st).feature("stat").set("mesh", new String[][]{{"geom1", "mesh2"}});
      } catch (Exception ignored) {
      }
      try {
        m.study(st).feature("stat").set("plot", "off");
      } catch (Exception ignored) {
      }
      try {
        m.study(st).run();
        String ds = datasetForStudy(st);
        double mx = evalMaxMises(m, ds, "mx_" + st + "_mesh2");
        logs.add("STUDY|" + st + "|status=ok|dataset=" + ds + "|max_mises=" + mx);
      } catch (Exception e) {
        logs.add("STUDY|" + st + "|status=fail|err=" + e.getMessage());
      }
    }

    try {
      m.save(MPH);
      logs.add("SAVED|" + MPH);
    } catch (IOException e) {
      logs.add("SAVE_FAIL|" + e.getMessage());
    }

    for (String line : logs) {
      System.out.println(line);
    }
  }
}
