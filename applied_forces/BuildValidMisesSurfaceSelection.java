import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;
import java.util.ArrayList;

public class BuildValidMisesSurfaceSelection {
  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/BuildValidMisesSurfaceSelection_Model.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.study("std1").run(); }
    catch (Exception e) { System.out.println("std1 failed: " + e.getMessage()); }

    int[] candidates = null;
    try {
      candidates = m.component("comp1").selection("mesh1_impmsh_mpart1_imp1___tracked_surface_stl").entities(2);
    } catch (Exception e) {}

    if (candidates == null || candidates.length == 0) {
      candidates = new int[1200];
      for (int i = 0; i < 1200; i++) candidates[i] = i + 1;
      System.out.println("candidate fallback range=1200");
    } else {
      System.out.println("candidate boundaries=" + candidates.length);
    }

    try { m.result().numerical().remove("mxchk"); } catch (Exception e) {}
    m.result().numerical().create("mxchk", "MaxSurface");
    m.result().numerical("mxchk").set("data", "dset6");
    m.result().numerical("mxchk").set("expr", new String[]{"solid.mises"});
    m.result().numerical("mxchk").set("unit", new String[]{"Pa"});

    ArrayList<Integer> valid = new ArrayList<Integer>();
    double globalMax = Double.NaN;

    for (int b : candidates) {
      try {
        m.result().numerical("mxchk").selection().geom("geom1", 2);
        m.result().numerical("mxchk").selection().set(new int[]{b});
        m.result().numerical("mxchk").setResult();
        double[][] r = m.result().numerical("mxchk").getReal();
        if (r != null && r.length > 0 && r[0].length > 0) {
          double v = r[0][0];
          if (!Double.isNaN(v) && !Double.isInfinite(v)) {
            valid.add(b);
            if (Double.isNaN(globalMax) || v > globalMax) globalMax = v;
          }
        }
      } catch (Exception e) {
        // skip non-meshed / invalid boundaries
      }
    }

    int[] vb = new int[valid.size()];
    for (int i = 0; i < valid.size(); i++) vb[i] = valid.get(i);

    System.out.println("valid_boundaries=" + vb.length);
    if (vb.length > 0) {
      StringBuilder sb = new StringBuilder();
      int n = Math.min(20, vb.length);
      for (int i = 0; i < n; i++) {
        if (i > 0) sb.append(',');
        sb.append(vb[i]);
      }
      System.out.println("valid_boundary_sample=" + sb.toString());
      System.out.println("max_surface_mises_valid=" + globalMax);
    }

    try { m.component("comp1").selection().remove("sel_mises_surface_valid"); } catch (Exception e) {}
    m.component("comp1").selection().create("sel_mises_surface_valid", "Explicit");
    m.component("comp1").selection("sel_mises_surface_valid").geom("geom1", 2);
    if (vb.length > 0) m.component("comp1").selection("sel_mises_surface_valid").set(vb);

    try { m.result().numerical().remove("mxsurf_valid"); } catch (Exception e) {}
    try {
      m.result().numerical().create("mxsurf_valid", "MaxSurface");
      m.result().numerical("mxsurf_valid").set("data", "dset6");
      m.result().numerical("mxsurf_valid").set("expr", new String[]{"solid.mises"});
      m.result().numerical("mxsurf_valid").set("unit", new String[]{"Pa"});
      m.result().numerical("mxsurf_valid").selection().named("sel_mises_surface_valid");
      m.result().numerical("mxsurf_valid").setResult();
      double[][] r = m.result().numerical("mxsurf_valid").getReal();
      double mx = (r!=null&&r.length>0&&r[0].length>0)?r[0][0]:Double.NaN;
      System.out.println("max_surface_mises_sel_mises_surface_valid=" + mx);
    } catch (Exception e) {
      System.out.println("mxsurf_valid failed: " + e.getMessage());
    }

    try {
      m.result("pg_vms_std1").set("data", "dset6");
      m.result("pg_vms_std1").feature("surf1").selection().named("sel_mises_surface_valid");
      m.result("pg_vms_std1").run();
      System.out.println("pg_vms_std1 bound_to_sel_mises_surface_valid=ok");
    } catch (Exception e) {
      System.out.println("pg_vms_std1 bind failed: " + e.getMessage());
    }

    try { m.save(out); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}
