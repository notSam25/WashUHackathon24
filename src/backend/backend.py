from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data (could be replaced with a database)
data = [
    {"id": 1, "name": "Item 1"},
    {"id": 2, "name": "Item 2"},
]

@app.route('/api/items', methods=['GET'])
def get_items():
    return jsonify(data)

@app.route('/api/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    item = next((item for item in data if item['id'] == item_id), None)
    if item:
        return jsonify(item)
    return jsonify({"error": "Item not found"}), 404

@app.route('/api/items', methods=['POST'])
def create_item():
    new_item = request.json
    new_item['id'] = len(data) + 1  # Simple ID generation
    data.append(new_item)
    return jsonify(new_item), 201

@app.route('/api/items/<int:item_id>', methods=['PUT'])
def update_item(item_id):
    item = next((item for item in data if item['id'] == item_id), None)
    if item:
        updated_data = request.json
        item.update(updated_data)
        return jsonify(item)
    return jsonify({"error": "Item not found"}), 404

@app.route('/api/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    global data
    data = [item for item in data if item['id'] != item_id]
    return jsonify({"message": "Item deleted"}), 204

if __name__ == '__main__':
    app.run(debug=True)
