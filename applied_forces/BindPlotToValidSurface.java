import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;
import java.util.ArrayList;

public class BindPlotToValidSurface {
  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/BindPlotToValidSurface_Model.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.study("std1").run(); }
    catch (Exception e) { System.out.println("std1 failed: " + e.getMessage()); }

    int[] candidates = null;
    try { candidates = m.component("comp1").selection("mesh1_impmsh_mpart1_imp1___tracked_surface_stl").entities(2); }
    catch (Exception e) {}
    if (candidates == null || candidates.length == 0) {
      candidates = new int[1200];
      for (int i = 0; i < 1200; i++) candidates[i] = i + 1;
    }

    try { m.result().numerical().remove("mxchk_bind"); } catch (Exception e) {}
    m.result().numerical().create("mxchk_bind", "MaxSurface");
    m.result().numerical("mxchk_bind").set("data", "dset6");
    m.result().numerical("mxchk_bind").set("expr", new String[]{"solid.mises"});
    m.result().numerical("mxchk_bind").set("unit", new String[]{"Pa"});

    ArrayList<Integer> valid = new ArrayList<Integer>();
    for (int b : candidates) {
      try {
        m.result().numerical("mxchk_bind").selection().geom("geom1", 2);
        m.result().numerical("mxchk_bind").selection().set(new int[]{b});
        m.result().numerical("mxchk_bind").setResult();
        double[][] r = m.result().numerical("mxchk_bind").getReal();
        if (r != null && r.length > 0 && r[0].length > 0) {
          double v = r[0][0];
          if (!Double.isNaN(v) && !Double.isInfinite(v)) valid.add(b);
        }
      } catch (Exception e) {}
    }

    int[] vb = new int[valid.size()];
    for (int i = 0; i < valid.size(); i++) vb[i] = valid.get(i);

    System.out.println("valid_boundaries_for_plot=" + vb.length);

    try { m.result().remove("pg_vms_std1_nonempty"); } catch (Exception e) {}
    try {
      m.result().create("pg_vms_std1_nonempty", "PlotGroup3D");
      m.result("pg_vms_std1_nonempty").label("Von Mises Stress Cloud (std1 nonempty)");
      m.result("pg_vms_std1_nonempty").set("data", "dset6");
      m.result("pg_vms_std1_nonempty").create("surf1", "Surface");
      m.result("pg_vms_std1_nonempty").feature("surf1").set("expr", "solid.mises");
      m.result("pg_vms_std1_nonempty").feature("surf1").set("unit", "Pa");
      m.result("pg_vms_std1_nonempty").feature("surf1").selection().geom("geom1", 2);
      m.result("pg_vms_std1_nonempty").feature("surf1").selection().set(vb);
      m.result("pg_vms_std1_nonempty").run();
      System.out.println("pg_vms_std1_nonempty=ok");
    } catch (Exception e) {
      System.out.println("pg_vms_std1_nonempty failed: " + e.getMessage());
    }

    try { m.result().numerical().remove("mxsurf_nonempty"); } catch (Exception e) {}
    try {
      m.result().numerical().create("mxsurf_nonempty", "MaxSurface");
      m.result().numerical("mxsurf_nonempty").set("data", "dset6");
      m.result().numerical("mxsurf_nonempty").set("expr", new String[]{"solid.mises"});
      m.result().numerical("mxsurf_nonempty").set("unit", new String[]{"Pa"});
      m.result().numerical("mxsurf_nonempty").selection().geom("geom1", 2);
      m.result().numerical("mxsurf_nonempty").selection().set(vb);
      m.result().numerical("mxsurf_nonempty").setResult();
      double[][] r = m.result().numerical("mxsurf_nonempty").getReal();
      double mx = (r!=null&&r.length>0&&r[0].length>0)?r[0][0]:Double.NaN;
      System.out.println("max_surface_mises_nonempty_plot_selection=" + mx);
    } catch (Exception e) {
      System.out.println("mxsurf_nonempty failed: " + e.getMessage());
    }

    try { m.save(out); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}
