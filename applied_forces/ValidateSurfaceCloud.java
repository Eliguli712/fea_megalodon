import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ValidateSurfaceCloud {
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

  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.study("std1").run(); } catch (Exception e) { System.out.println("std1 failed: " + e.getMessage()); }

    int[] dom = keptDomains();

    try { m.result().dataset().remove("dset_std1_sel"); } catch (Exception e) {}
    try {
      m.result().dataset().create("dset_std1_sel", "Selection");
      m.result().dataset("dset_std1_sel").set("data", "dset6");
      m.result().dataset("dset_std1_sel").selection().geom("geom1", 3);
      m.result().dataset("dset_std1_sel").selection().set(dom);
      System.out.println("dataset dset_std1_sel created");
    } catch (Exception e) {
      System.out.println("dataset creation failed: " + e.getMessage());
    }

    try { m.result().numerical().remove("mxsurf_sel"); } catch (Exception e) {}
    try {
      m.result().numerical().create("mxsurf_sel", "MaxSurface");
      m.result().numerical("mxsurf_sel").set("data", "dset_std1_sel");
      m.result().numerical("mxsurf_sel").set("expr", new String[]{"solid.mises"});
      m.result().numerical("mxsurf_sel").set("unit", new String[]{"Pa"});
      m.result().numerical("mxsurf_sel").selection().all();
      m.result().numerical("mxsurf_sel").setResult();
      double[][] r = m.result().numerical("mxsurf_sel").getReal();
      double mx = (r!=null&&r.length>0&&r[0].length>0)?r[0][0]:Double.NaN;
      System.out.println("max_surface_mises_selected_dataset=" + mx);
    } catch (Exception e) {
      System.out.println("surface eval failed: " + e.getMessage());
    }

    try { m.result().remove("pg_vms_std1_sel"); } catch (Exception e) {}
    try {
      m.result().create("pg_vms_std1_sel", "PlotGroup3D");
      m.result("pg_vms_std1_sel").label("Von Mises Stress Cloud (std1 selected)");
      m.result("pg_vms_std1_sel").set("data", "dset_std1_sel");
      m.result("pg_vms_std1_sel").create("surf1", "Surface");
      m.result("pg_vms_std1_sel").feature("surf1").set("expr", "solid.mises");
      m.result("pg_vms_std1_sel").feature("surf1").set("unit", "Pa");
      m.result("pg_vms_std1_sel").run();
      System.out.println("pg_vms_std1_sel run ok");
    } catch (Exception e) {
      System.out.println("plot run failed: " + e.getMessage());
    }

    try { m.save("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ValidateSurfaceCloud_Model.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}
