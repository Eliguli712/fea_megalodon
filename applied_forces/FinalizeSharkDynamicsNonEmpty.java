import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;
import java.util.ArrayList;

public class FinalizeSharkDynamicsNonEmpty {
  private static int[] keptDomains() {
    int[] rem = new int[]{2,5,6,25,28,46,48,51,62,84,91,98,100,106,110,116,121,127,131,135,152,165,182};
    boolean[] keep = new boolean[184];
    for (int i=1;i<=183;i++) keep[i] = true;
    for (int r: rem) keep[r] = false;
    int n=0; for (int i=1;i<=183;i++) if (keep[i]) n++;
    int[] out = new int[n]; int k=0;
    for (int i=1;i<=183;i++) if (keep[i]) out[k++]=i;
    return out;
  }

  private static double evalMaxSelected(Model m, String tag, String dset, int[] dom) {
    try { m.result().numerical().remove(tag); } catch (Exception e) {}
    try {
      m.result().numerical().create(tag, "MaxVolume");
      m.result().numerical(tag).set("expr", new String[]{"solid.mises"});
      m.result().numerical(tag).set("unit", new String[]{"Pa"});
      m.result().numerical(tag).set("data", dset);
      m.result().numerical(tag).selection().geom("geom1", 3);
      m.result().numerical(tag).selection().set(dom);
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      return (r!=null&&r.length>0&&r[0].length>0)?r[0][0]:Double.NaN;
    } catch (Exception e) {
      return Double.NaN;
    }
  }

  public static void main(String[] args) {
    String path = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    Model m;
    try { m = ModelUtil.load("Model", path); }
    catch (IOException e) { throw new RuntimeException(e); }

    int[] dom = keptDomains();

    try { m.study("std1").run(); System.out.println("std1=ok"); }
    catch (Exception e) { System.out.println("std1=fail: " + e.getMessage()); }

    double mxStd1 = evalMaxSelected(m, "mx_std1_selected_final", "dset6", dom);
    System.out.println("std1_max_mises_selected=" + mxStd1);

    int[] candidates = null;
    try { candidates = m.component("comp1").selection("mesh1_impmsh_mpart1_imp1___tracked_surface_stl").entities(2); }
    catch (Exception e) {}
    if (candidates == null || candidates.length == 0) {
      candidates = new int[1200];
      for (int i = 0; i < 1200; i++) candidates[i] = i + 1;
    }

    try { m.result().numerical().remove("mx_surf_probe"); } catch (Exception e) {}
    m.result().numerical().create("mx_surf_probe", "MaxSurface");
    m.result().numerical("mx_surf_probe").set("data", "dset6");
    m.result().numerical("mx_surf_probe").set("expr", new String[]{"solid.mises"});
    m.result().numerical("mx_surf_probe").set("unit", new String[]{"Pa"});

    ArrayList<Integer> valid = new ArrayList<Integer>();
    double maxSurf = Double.NaN;
    for (int b : candidates) {
      try {
        m.result().numerical("mx_surf_probe").selection().geom("geom1", 2);
        m.result().numerical("mx_surf_probe").selection().set(new int[]{b});
        m.result().numerical("mx_surf_probe").setResult();
        double[][] r = m.result().numerical("mx_surf_probe").getReal();
        if (r != null && r.length > 0 && r[0].length > 0) {
          double v = r[0][0];
          if (!Double.isNaN(v) && !Double.isInfinite(v)) {
            valid.add(b);
            if (Double.isNaN(maxSurf) || v > maxSurf) maxSurf = v;
          }
        }
      } catch (Exception e) {}
    }
    System.out.println("surface_valid_boundary_count=" + valid.size());
    System.out.println("surface_max_mises_nonempty=" + maxSurf);

    try { m.study("std_nh").run(); System.out.println("std_nh=ok"); }
    catch (Exception e) { System.out.println("std_nh=fail: " + e.getMessage()); }
    System.out.println("std_nh_max_mises_selected=" + evalMaxSelected(m, "mx_nh_selected_final", "dset1", dom));

    try { m.study("std_og").run(); System.out.println("std_og=ok"); }
    catch (Exception e) { System.out.println("std_og=fail: " + e.getMessage()); }
    System.out.println("std_og_max_mises_selected=" + evalMaxSelected(m, "mx_og_selected_final", "dset2", dom));

    try { m.result("pg_vms_std1").run(); System.out.println("pg_vms_std1=ok"); }
    catch (Exception e) { System.out.println("pg_vms_std1=fail: " + e.getMessage()); }

    try { m.save(path); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}
