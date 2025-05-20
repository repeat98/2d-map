// ColumnContextMenu.jsx

import React from 'react';
import PropTypes from 'prop-types';
import './ColumnContextMenu.scss';

const ColumnContextMenu = ({ x, y, column, visibleColumns, onToggleColumn }) => {
  return (
    <div className="column-context-menu" style={{ top: y, left: x }}>
      {visibleColumns.map((col) => (
        <div key={col.name} className="menu-item">
          <label>
            <input
              type="checkbox"
              checked={col.visible}
              onChange={(e) => onToggleColumn(col.name, e.target.checked)}
            />
            {col.name}
          </label>
        </div>
      ))}
    </div>
  );
};

ColumnContextMenu.propTypes = {
  x: PropTypes.number.isRequired,
  y: PropTypes.number.isRequired,
  column: PropTypes.string.isRequired,
  visibleColumns: PropTypes.arrayOf(
    PropTypes.shape({
      name: PropTypes.string.isRequired,
      visible: PropTypes.bool.isRequired,
    })
  ).isRequired,
  onToggleColumn: PropTypes.func.isRequired,
};

export default ColumnContextMenu;