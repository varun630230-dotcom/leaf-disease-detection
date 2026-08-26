import { useState, useMemo } from 'react';
import { ClassPerformance } from '../types';
import { ArrowUpDown, Table as TableIcon, Search } from 'lucide-react';

interface Props {
  data: ClassPerformance[];
}

type SortField = 'plant' | 'disease' | 'precision' | 'recall' | 'f1' | 'support';
type SortOrder = 'asc' | 'desc';

export default function MetricsTable({ data }: Props) {
  const [sortField, setSortField] = useState<SortField>('f1');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [searchTerm, setSearchTerm] = useState('');

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder(field === 'plant' || field === 'disease' ? 'asc' : 'desc');
    }
  };

  const filteredAndSorted = useMemo(() => {
    return [...data]
      .filter(item => {
        const query = searchTerm.toLowerCase();
        return (
          item.plant.toLowerCase().includes(query) ||
          item.disease.toLowerCase().includes(query) ||
          item.className.toLowerCase().includes(query)
        );
      })
      .sort((a, b) => {
        let valA = a[sortField];
        let valB = b[sortField];

        if (typeof valA === 'string') {
          valA = (valA as string).toLowerCase();
          valB = (valB as string).toLowerCase();
        }

        if (valA < valB) return sortOrder === 'asc' ? -1 : 1;
        if (valA > valB) return sortOrder === 'asc' ? 1 : -1;
        return 0;
      });
  }, [data, sortField, sortOrder, searchTerm]);

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
      {/* Header & Filter Search */}
      <div className="p-4 sm:px-6 sm:py-4 border-b border-slate-200 bg-slate-50/70 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <TableIcon className="w-4 h-4 text-emerald-600" />
          <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
            Per-Class Validation Performance ({data.length} Classes)
          </h3>
        </div>

        <div className="relative w-full sm:w-64">
          <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search plant or disease..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 text-xs rounded-md border border-slate-200 focus:outline-none focus:ring-1 focus:ring-emerald-500 bg-white"
          />
        </div>
      </div>

      {/* Responsive Table */}
      <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-50 text-slate-500 uppercase font-semibold border-b border-slate-200 sticky top-0 z-10">
            <tr>
              <th
                onClick={() => handleSort('plant')}
                className="py-3 px-5 cursor-pointer hover:text-slate-900 select-none"
              >
                <div className="flex items-center gap-1">
                  <span>Plant</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-400" />
                </div>
              </th>
              <th
                onClick={() => handleSort('disease')}
                className="py-3 px-5 cursor-pointer hover:text-slate-900 select-none"
              >
                <div className="flex items-center gap-1">
                  <span>Condition / Disease</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-400" />
                </div>
              </th>
              <th
                onClick={() => handleSort('precision')}
                className="py-3 px-5 cursor-pointer hover:text-slate-900 select-none text-right"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>Precision</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-400" />
                </div>
              </th>
              <th
                onClick={() => handleSort('recall')}
                className="py-3 px-5 cursor-pointer hover:text-slate-900 select-none text-right"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>Recall</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-400" />
                </div>
              </th>
              <th
                onClick={() => handleSort('f1')}
                className="py-3 px-5 cursor-pointer hover:text-slate-900 select-none text-right"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>F1 Score</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-400" />
                </div>
              </th>
              <th
                onClick={() => handleSort('support')}
                className="py-3 px-5 cursor-pointer hover:text-slate-900 select-none text-right"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>Support</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-400" />
                </div>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {filteredAndSorted.length > 0 ? (
              filteredAndSorted.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-50/70 transition-colors">
                  <td className="py-2.5 px-5 font-semibold text-slate-900">{row.plant}</td>
                  <td className="py-2.5 px-5">
                    {row.isHealthy ? (
                      <span className="text-emerald-700 font-medium">Healthy</span>
                    ) : (
                      <span className="text-slate-700">{row.disease}</span>
                    )}
                  </td>
                  <td className="py-2.5 px-5 text-right font-mono text-slate-600">
                    {(row.precision * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-5 text-right font-mono text-slate-600">
                    {(row.recall * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-5 text-right font-mono font-semibold text-slate-900">
                    {(row.f1 * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-5 text-right font-mono text-slate-400">
                    {row.support}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={6} className="py-8 text-center text-slate-400">
                  No classes match your search.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
