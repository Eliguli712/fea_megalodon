import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;
import java.util.Arrays;

public class ProbeMesh2ExcludeDomain {
  private static void p(String s) { System.out.println(s); }

  private static int[] domainSetExcluding(int n, int bad) {
    int[] a = new int[n - 1];
    int k = 0;
    for (int i = 1; i <= n; i++) if (i != bad) a[k++] = i;
    return a;
  }

  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeMesh2ExcludeDomain_Model.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    try {
      try { m.component("comp1").mesh("mesh1").feature().remove("impmsh"); } catch (Exception e) {}
      m.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "sequence");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("sequence", "mpart1");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("buildsource", "on");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("domelemsequence", "on");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("unmesheddom", "on");
      m.component("comp1").mesh("mesh1").run();
      p("mesh1 import run ok");
    } catch (Exception e) {
      p("mesh1 import run failed: " + e.getMessage());
      e.printStackTrace();
    }

    try { m.component("comp1").mesh().remove("mesh2"); } catch (Exception e) {}
    m.component("comp1").mesh().create("mesh2", "geom1");
    m.component("comp1").mesh("mesh2").automatic(true);
    m.component("comp1").mesh("mesh2").autoMeshSize(5);

    try {
      String[] tags = m.component("comp1").mesh("mesh2").feature().tags();
      p("mesh2 features: " + Arrays.toString(tags));
      int[] dom = domainSetExcluding(183, 5);
      boolean setAny = false;
      for (String t : tags) {
        try {
          m.component("comp1").mesh("mesh2").feature(t).selection().geom("geom1", 3);
          m.component("comp1").mesh("mesh2").feature(t).selection().set(dom);
          p("set domain selection on feature " + t + " with " + dom.length + " domains");
          setAny = true;
        } catch (Exception ex) {
          p("feature " + t + " selection set failed: " + ex.getMessage());
        }
      }
      if (!setAny) p("no selectable mesh2 feature found");
    } catch (Exception e) {
      p("mesh2 feature inspection failed: " + e.getMessage());
    }

    try {
      m.component("comp1").mesh("mesh2").run();
      p("mesh2 run ok");
    } catch (Exception e) {
      p("mesh2 run failed: " + e.getMessage());
      e.printStackTrace();
    }

    try {
      // Restrict solid physics to same meshed domains.
      int[] dom = domainSetExcluding(183, 5);
      m.component("comp1").physics("solid").selection().geom("geom1", 3);
      m.component("comp1").physics("solid").selection().set(dom);
      p("solid selection set to 182 domains");
    } catch (Exception e) {
      p("solid selection set failed: " + e.getMessage());
    }

    try {
      m.study("std1").feature("stat").set("mesh", new String[][]{{"geom1","mesh2"}});
      p("std1 mesh switched to mesh2");
    } catch (Exception e) {
      p("std1 mesh switch failed: " + e.getMessage());
    }

    try {
      m.study("std1").run();
      p("std1 run ok");
    } catch (Exception e) {
      p("std1 run failed: " + e.getMessage());
      e.printStackTrace();
    }

    try {
      try { m.result().numerical().remove("max_probe2"); } catch (Exception e) {}
      m.result().numerical().create("max_probe2", "MaxSurface");
      m.result().numerical("max_probe2").set("expr", new String[]{"solid.mises"});
      m.result().numerical("max_probe2").set("unit", new String[]{"Pa"});
      m.result().numerical("max_probe2").setResult();
      double[][] v = m.result().numerical("max_probe2").getReal();
      if (v != null && v.length > 0 && v[0].length > 0) p("max solid.mises = " + v[0][0]);
      else p("max solid.mises unavailable");
    } catch (Exception e) {
      p("mises eval failed: " + e.getMessage());
      e.printStackTrace();
    }

    try { m.save(out); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}
