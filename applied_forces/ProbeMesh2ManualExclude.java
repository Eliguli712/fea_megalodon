import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeMesh2ManualExclude {
  private static void p(String s) { System.out.println(s); }

  private static int[] domainSetExcluding(int n, int bad) {
    int[] a = new int[n - 1];
    int k = 0;
    for (int i = 1; i <= n; i++) if (i != bad) a[k++] = i;
    return a;
  }

  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeMesh2ManualExclude_Model.mph";
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
    m.component("comp1").mesh("mesh2").automatic(false);

    try { m.component("comp1").mesh("mesh2").feature().remove("size1"); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh2").feature().remove("ftet1"); } catch (Exception e) {}

    m.component("comp1").mesh("mesh2").feature().create("size1", "Size");
    m.component("comp1").mesh("mesh2").feature("size1").set("hauto", 5);

    m.component("comp1").mesh("mesh2").feature().create("ftet1", "FreeTet");

    int[] dom = domainSetExcluding(183, 5);
    try {
      m.component("comp1").mesh("mesh2").feature("size1").selection().geom("geom1", 3);
      m.component("comp1").mesh("mesh2").feature("size1").selection().set(dom);
      p("size1 domain selection set: " + dom.length);
    } catch (Exception e) {
      p("size1 selection failed: " + e.getMessage());
    }

    try {
      m.component("comp1").mesh("mesh2").feature("ftet1").selection().geom("geom1", 3);
      m.component("comp1").mesh("mesh2").feature("ftet1").selection().set(dom);
      p("ftet1 domain selection set: " + dom.length);
    } catch (Exception e) {
      p("ftet1 selection failed: " + e.getMessage());
    }

    try {
      m.component("comp1").mesh("mesh2").run();
      p("mesh2 run ok");
    } catch (Exception e) {
      p("mesh2 run failed: " + e.getMessage());
      e.printStackTrace();
    }

    try {
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
      try { m.result().numerical().remove("max_probe3"); } catch (Exception e) {}
      m.result().numerical().create("max_probe3", "MaxSurface");
      m.result().numerical("max_probe3").set("expr", new String[]{"solid.mises"});
      m.result().numerical("max_probe3").set("unit", new String[]{"Pa"});
      m.result().numerical("max_probe3").setResult();
      double[][] v = m.result().numerical("max_probe3").getReal();
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
